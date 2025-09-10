# scripts/train_drl_agent.py

import heapq
import logging
import sys
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import wrench.schedulers as schedulers  # <--- UPDATED IMPORT
import yaml
from gymnasium import spaces
from wrench.events import EventType

sys.path.append('.')

from src.ai_schedulers import WASSDRLScheduler
from src.drl_agent import DQNAgent, SchedulingState, SchedulingAction
from src.factory import PlatformFactory, WorkflowFactory
from src.utils import get_logger
from wass_wrench_simulator import WassWrenchSimulator

logger = get_logger(__name__, logging.INFO)

class WassEnv(gym.Env):
    """A Gym environment for training a WASS DRL agent."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        self.platform_factory = PlatformFactory(config['platform'])
        self.platform = self.platform_factory.get_platform()
        self.node_names = list(self.platform.get_nodes().keys())
        
        self.workflow_factory = WorkflowFactory(config['workflow'])
        
        self.num_nodes = len(self.node_names)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        state_size = 3 + 3 * self.num_nodes
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

        self.baseline_scheduler = schedulers.HeftScheduler(platform=self.platform)
        self.baseline_makespan = self._get_baseline_makespan()
        
        self.simulator: Optional[WassWrenchSimulator] = None
        self.drl_scheduler_in_training: Optional['DRLSchedulerForEnv'] = None

    def _get_baseline_makespan(self) -> float:
        """Run a simulation with HEFT to get a reference makespan."""
        logger.info("Calculating baseline makespan with HEFT...")
        workflow = self.workflow_factory.get_workflow()
        heft_sim = schedulers.Simulation(self.baseline_scheduler, self.platform, workflow)
        heft_sim.run()
        logger.info(f"Baseline HEFT makespan: {heft_sim.time:.2f}")
        return heft_sim.time

    def reset(self, *, seed=None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        
        workflow = self.workflow_factory.get_workflow()
        self.drl_scheduler_in_training = DRLSchedulerForEnv(self.node_names)
        self.simulator = WassWrenchSimulator(self.drl_scheduler_in_training, self.platform, workflow)
        self.simulator._initialize_simulation()
        
        if not self.simulator.event_queue:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        first_event = self.simulator.event_queue[0]
        first_task = first_event.event_data
        
        self.drl_scheduler_in_training.set_next_task(first_task)
        state = self.drl_scheduler_in_training._extract_features(first_task, self.simulator)
        
        return state.vector, {}

    def step(self, action: SchedulingAction) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        chosen_node_name = self.node_names[action]
        self.drl_scheduler_in_training.set_next_action(chosen_node_name)
        
        ready_event = heapq.heappop(self.simulator.event_queue)
        self.simulator._handle_task_ready(ready_event.time, ready_event.event_data)

        next_state_vec = np.zeros(self.observation_space.shape, dtype=np.float32)
        done = False
        
        while self.simulator.event_queue:
            event = self.simulator.event_queue[0]
            if event.event_type == EventType.TASK_READY:
                next_task = event.event_data
                self.drl_scheduler_in_training.set_next_task(next_task)
                next_state = self.drl_scheduler_in_training._extract_features(next_task, self.simulator)
                next_state_vec = next_state.vector
                break
            else:
                event = heapq.heappop(self.simulator.event_queue)
                self.simulator._handle_task_finish(event.time, event.event_data)
        
        reward = 0.0
        if not self.simulator.event_queue:
            done = True
            final_makespan = self.simulator._calculate_makespan()
            reward = self.baseline_makespan - final_makespan
            
        return next_state_vec, reward, done, False, {}

class DRLSchedulerForEnv(WASSDRLScheduler):
    def __init__(self, node_names: List[str]):
        super().__init__(drl_agent=None, node_names=node_names)
        self.current_task: Optional[w.Task] = None
        self.next_action_node: Optional[str] = None

    def set_next_task(self, task: w.Task):
        self.current_task = task
        
    def set_next_action(self, node_name: str):
        self.next_action_node = node_name

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> Dict:
        if not self.current_task or not self.next_action_node:
            raise ValueError("set_next_task and set_next_action must be called before schedule")
        
        decision = {self.next_action_node: self.current_task}
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

    env = WassEnv(config)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent_config = config['drl']['agent']
    agent = DQNAgent(state_size, action_size, seed=agent_config.get('seed', 0))

    n_episodes = agent_config.get('n_episodes', 1000)
    eps_start = agent_config.get('eps_start', 1.0)
    eps_end = agent_config.get('eps_end', 0.01)
    eps_decay = agent_config.get('eps_decay', 0.995)
    
    eps = eps_start
    scores = []
    
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        done = False
        while not done:
            action = agent.act(SchedulingState(state), eps)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(SchedulingState(state), action, reward, SchedulingState(next_state), done)
            state = next_state
            score += reward
            if done or truncated:
                break
        
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        
        print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores[-100:]):.2f}")

    model_path = config['drl']['model_path']
    agent.save(model_path)
    logger.info(f"\nDRL agent model saved to {model_path}")