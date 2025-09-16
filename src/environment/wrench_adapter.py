import wrench
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Callable, Dict, List
import xml.etree.ElementTree as ET
import threading

class PythonRLScheduler:
    """A custom Python scheduler that interacts with the WRENCH core."""
    def __init__(self, compute_hosts: List[str]):
        self.compute_hosts = compute_hosts
        self.waiting_job = None
        self.decision = None
        self.decision_needed_event = threading.Event()
        self.decision_made_event = threading.Event()

    def make_scheduling_decision(self, job: wrench.StandardJob, available_resources: List[str]):
        """This method is called by the WRENCH simulation thread."""
        self.waiting_job = job
        self.decision_made_event.clear()
        self.decision_needed_event.set()
        self.decision_made_event.wait()
        return self.decision

    def get_waiting_job(self):
        """Called by the main thread to get the job that needs scheduling."""
        self.decision_needed_event.wait()
        return self.waiting_job

    def set_decision(self, host_id: int):
        """Called by the main thread to provide the DRL agent's action."""
        if 0 <= host_id < len(self.compute_hosts):
            self.decision = self.compute_hosts[host_id]
        else:
            self.decision = self.compute_hosts[0]
        self.decision_needed_event.clear()
        self.decision_made_event.set()

def parse_platform(platform_file: str) -> (str, Dict[str, Dict]):
    """Parses a platform XML file to extract properties."""
    host_properties = {}
    controller_host = "ControllerHost"
    try:
        tree = ET.parse(platform_file)
        root = tree.getroot()
        first_host_elem = root.find(".//host")
        if first_host_elem is not None:
            controller_host = first_host_elem.get("id")
        for host_elem in root.findall(".//host"):
            host_name = host_elem.get("id")
            if host_name != controller_host:
                speed_str = host_elem.get("speed", "0Gf")
                speed = float(speed_str.replace("Gf", "").strip())
                host_properties[host_name] = {'speed': speed}
    except Exception as e:
        print(f"[ERROR] Failed to parse platform file {platform_file}: {e}"); raise
    return controller_host, host_properties

class WrenchEnv(gym.Env):
    """Final adapter for the WRENCH simulator using a custom Python scheduler."""
    def __init__(self, platform_file: str, workflow_file: str, feature_extractor: Callable):
        super(WrenchEnv, self).__init__()
        self.platform_file = platform_file
        self.workflow_file = workflow_file
        self.feature_extractor = feature_extractor
        
        with open(self.platform_file, "r") as f:
            self.platform_xml_string = f.read()
        
        self.controller_host, self.host_properties = parse_platform(platform_file)
        self.compute_host_names = list(self.host_properties.keys())

        # We must create the services BEFORE the simulation object
        self.compute_service = wrench.BareMetalComputeService(self.controller_host, self.compute_host_names)
        self.drl_scheduler = PythonRLScheduler(self.compute_host_names)
        
        self._initialize_spaces()
        self.simulation = None
        self.workflow = None

    def _initialize_spaces(self):
        self.action_space = spaces.Discrete(len(self.compute_host_names))
        sample_state = self.feature_extractor(None, self.compute_host_names, self.host_properties)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=sample_state.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pass the services to the simulation constructor
        self.simulation = wrench.Simulation(self.compute_service, self.drl_scheduler.make_scheduling_decision)
        
        self.workflow = self.simulation.create_workflow_from_json(self.workflow_file)
        job = self.simulation.create_standard_job(self.workflow.get_tasks())
        self.compute_service.submit_job(job)

        initial_state = self._get_state()
        return initial_state, {}

    def step(self, action: int):
        self.drl_scheduler.set_decision(action)
        
        # A simple check for termination
        done = all(t.is_finished() for t in self.workflow.get_tasks())
        
        state = self._get_state()
        
        reward = 0.0
        makespan = self.simulation.get_simulated_time() if done else 0.0
        info = {'makespan': makespan}
        terminated = done
        truncated = False

        return state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        job = self.drl_scheduler.get_waiting_job()
        if job is None or not job.get_tasks() or all(t.is_finished() for t in job.get_tasks()):
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        task_to_schedule = next((t for t in job.get_tasks() if not t.is_submitted()), None)
        
        return self.feature_extractor(task_to_schedule, self.compute_host_names, self.host_properties)

    def close(self):
        pass