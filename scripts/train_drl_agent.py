import logging
import sys
from typing import Dict, Tuple, Optional

import gym
import numpy as np
import yaml
from gym import spaces

# Add the project root to the Python path
sys.path.append('.')

from src.drl_agent import DQNAgent, SchedulingState, SchedulingAction
from src.factory import PlatformFactory, WorkflowFactory, SchedulerFactory
from src.utils import get_logger
from wass_wrench_simulator import WassWrenchSimulator

logger = get_logger(__name__, logging.INFO)


class WassEnv(gym.Env):
    """A Gym environment for training a WASS DRL agent."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Factories
        self.platform_factory = PlatformFactory(config['platform'])
        self.platform = self.platform_factory.get_platform()
        self.node_names = list(self.platform.get_nodes().keys())
        
        self.workflow_factory = WorkflowFactory(config['workflow'])
        self.workflow = self.workflow_factory.get_workflow()

        # Define action and observation space
        self.num_nodes = len(self.node_names)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # The observation space needs to match the features from ai_schedulers
        # It's a flattened vector of task features + per-node features
        # Task features: computation_size, num_parents, num_children (3)
        # Per-node features: speed, earliest_finish_time, total_d_time (3)
        # Total size = 3 + 3 * num_nodes
        state_size = 3 + 3 * self.num_nodes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

        # Create a baseline scheduler for reward calculation
        self.scheduler_factory = SchedulerFactory(config, self.node_names)
        self.baseline_scheduler = self.scheduler_factory.get_scheduler("HEFT") # Using HEFT as baseline
        self.baseline_makespan = self._get_baseline_makespan()
        
        # Simulation state will be managed by an internal simulator instance
        self.simulator: Optional[WassWrenchSimulator] = None
        self.drl_scheduler_in_training: Optional['DRLSchedulerForEnv'] = None

    def _get_baseline_makespan(self) -> float:
        """Run a simulation with a baseline scheduler to get a reference makespan."""
        logger.info("Calculating baseline makespan with HEFT...")
        # HEFT needs a standard Wrench simulator
        heft_sim = schedulers.WrenchSimulation(self.baseline_scheduler, self.platform, self.workflow)
        heft_sim.run()
        logger.info(f"Baseline HEFT makespan: {heft_sim.time:.2f}")
        return heft_sim.time

    def reset(self) -> SchedulingState:
        """Resets the environment to an initial state."""
        # The DRL scheduler is re-created for each episode
        self.drl_scheduler_in_training = DRLSchedulerForEnv(self.node_names)
        
        # The simulator is also re-created
        self.simulator = WassWrenchSimulator(self.drl_scheduler_in_training, self.platform, self.workflow)
        self.simulator._initialize_simulation() # Start the simulation process
        
        # The first state is determined by the first ready task
        first_event = heapq.heappop(self.simulator.event_queue)
        first_task = first_event.event_data
        
        # The scheduler needs to be aware of the task it's deciding on
        self.drl_scheduler_in_training.set_next_task(first_task)

        # Extract features for the first state
        state = self.drl_scheduler_in_training._extract_features(first_task, self.simulator)
        return state.vector

    def step(self, action: SchedulingAction) -> Tuple[SchedulingState, float, bool, Dict]:
        """Execute one time step within the environment."""
        # The action determines the node for the current task
        chosen_node_name = self.node_names[action]
        current_task = self.drl_scheduler_in_training.current_task
        
        # Manually perform the scheduling step inside the simulator logic
        self.simulator._handle_task_ready(self.simulator.event_queue[0].time if self.simulator.event_queue else 0, current_task)

        # Run the simulation until the next scheduling decision is needed
        next_state_vec = None
        done = False
        
        while self.simulator.event_queue:
            event = heapq.heappop(self.simulator.event_queue)
            event_type = event.event_type
            
            if event_type == EventType.TASK_FINISH:
                self.simulator._handle_task_finish(event.time, event.event_data)
            elif event_type == EventType.TASK_READY:
                # This is the next decision point
                next_task = event.event_data
                self.drl_scheduler_in_training.set_next_task(next_task)
                next_state = self.drl_scheduler_in_training._extract_features(next_task, self.simulator)
                next_state_vec = next_state.vector
                # Push the event back so the next step can process it
                heapq.heappush(self.simulator.event_queue, event)
                break # Exit the loop to wait for the next action

        if not self.simulator.event_queue:
            # Simulation is over
            done = True
            final_makespan = self.simulator._calculate_makespan()
            # Reward is based on improvement over baseline
            reward = self.baseline_makespan - final_makespan
            # Return a zero vector as the next state
            next_state_vec = np.zeros(self.observation_space.shape)
        else:
            # Intermediate reward can be 0, or something more complex
            reward = 0.0
            
        return next_state_vec, reward, done, {}

    def render(self, mode='human'):
        pass


class DRLSchedulerForEnv(WASSDRLScheduler):
    """A special scheduler for the Gym env that pauses to get actions."""
    def __init__(self, node_names: List[str]):
        # This scheduler doesn't need a real agent, it just extracts features
        # and waits for the `schedule` method to be called with an action.
        super().__init__(drl_agent=None, node_names=node_names)
        self.current_task: Optional[w.Task] = None
        self.next_action_node: Optional[str] = None

    def set_next_task(self, task: w.Task):
        self.current_task = task
        
    def set_next_action(self, node_name: str):
        self.next_action_node = node_name

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> SchedulingDecision:
        # In the env, the action is provided externally via set_next_action
        if not self.current_task or not self.next_action_node:
            raise ValueError("set_next_task and set_next_action must be called before schedule")
        
        decision = {self.next_action_node: self.current_task}
        # Reset for the next turn
        self.current_task = None
        self.next_action_node = None
        return decision

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/train_drl_agent.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    env = WassEnv(config['environment'])
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent_config = config['drl']['agent']
    agent = DQNAgent(state_size, action_size, seed=agent_config.get('seed', 0))

    # Training loop
    n_episodes = agent_config.get('n_episodes', 1000)
    max_t = agent_config.get('max_t', 1000) # max number of steps per episode
    
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(SchedulingState(state))
            next_state, reward, done, _ = env.step(action)
            agent.step(SchedulingState(state), action, reward, SchedulingState(next_state), done)
            state = next_state
            score += reward
            if done:
                break
        
        logger.info(f"Episode {i_episode}\tScore: {score:.2f}")

    # Save the trained model
    model_path = config['drl']['model_path']
    agent.save(model_path)
    logger.info(f"DRL agent model saved to {model_path}")