# src/factory.py

import logging
from typing import Dict, List

import torch
import wrench.schedulers as schedulers  # <--- UPDATED IMPORT
import wrench.workflows as w
from wrench.platforms import Platform

from src.ai_schedulers import (WASSScheduler, WASSDRLScheduler,
                               WASSRAGScheduler)
from src.drl_agent import DQNAgent
from src.performance_predictor import PerformancePredictor
from src.utils import get_logger

logger = get_logger(__name__, logging.INFO)

class PlatformFactory:
    """Factory to create simulation platforms."""
    def __init__(self, config: Dict):
        self.config = config

    def get_platform(self) -> Platform:
        """Builds and returns a WRENCH platform."""
        platform = Platform(self.config['platform_file'])
        return platform

class WorkflowFactory:
    """Factory to create workflows."""
    def __init__(self, config: Dict):
        self.config = config

    def get_workflow(self) -> w.Workflow:
        """Builds and returns a WRENCH workflow."""
        workflow = w.Workflow(self.config['workflow_file'], self.config['input_size'])
        return workflow

class SchedulerFactory:
    """Factory to create various schedulers."""
    def __init__(self, config: Dict, node_names: List[str]):
        self.config = config
        self.node_names = node_names
        self.platform = PlatformFactory(config['platform']).get_platform()
        self.drl_agent = None
        self.predictor = None

    def _get_drl_agent(self) -> DQNAgent:
        """Lazily loads and returns the DRL agent."""
        if self.drl_agent is None:
            drl_config = self.config['drl']
            agent_config = drl_config['agent']
            state_size = 3 + 3 * len(self.node_names)
            action_size = len(self.node_names)
            
            self.drl_agent = DQNAgent(state_size, action_size, seed=agent_config.get('seed', 0))
            self.drl_agent.load(drl_config['model_path'])
            logger.info("Loaded trained DRL agent from %s", drl_config['model_path'])
        return self.drl_agent

    def _get_predictor(self) -> PerformancePredictor:
        """Lazily loads and returns the performance predictor."""
        if self.predictor is None:
            predictor_config = self.config['predictor']
            self.predictor = PerformancePredictor(
                model_type=predictor_config.get('model_type', 'RandomForest'),
                model_path=predictor_config['model_path']
            )
            logger.info("Loaded trained performance predictor from %s", predictor_config['model_path'])
        return self.predictor

    def get_scheduler(self, name: str) -> WASSScheduler:
        """Returns a scheduler instance based on its name."""
        if name == "HEFT":
            # The standard Wrench schedulers have a different API
            # We wrap it for compatibility with our simulator
            return HeuristicWrapper(schedulers.HeftScheduler(self.platform))
        elif name == "FIFO":
            return HeuristicWrapper(schedulers.FifoScheduler(self.platform))
        elif name == "WASS (Heuristic)":
            # Assuming FIFO is the heuristic for WASS
            return HeuristicWrapper(schedulers.FifoScheduler(self.platform))
        elif name == "WASS-DRL (w/o RAG)":
            return WASSDRLScheduler(self._get_drl_agent(), self.node_names)
        elif name == "WASS-RAG":
            rag_scheduler = WASSRAGScheduler(self._get_drl_agent(), self.node_names, self._get_predictor())
            # Here you would fit the scaler/vectorizer with data
            # This part is simplified for now.
            logger.info("Created WASS-RAG scheduler.")
            return rag_scheduler
        else:
            raise ValueError(f"Unknown scheduler name: {name}")


class HeuristicWrapper(WASSScheduler):
    """A wrapper to make standard Wrench schedulers compatible with our simulator."""
    def __init__(self, heuristic_scheduler):
        self.scheduler = heuristic_scheduler

    def schedule(self, ready_tasks: List[w.Task], simulation: 'WassWrenchSimulator') -> Dict:
        # Standard schedulers decide for all ready tasks at once.
        # Our simulator calls for one task at a time.
        if not ready_tasks:
            return {}
        task_to_schedule = ready_tasks[0]
        
        # The standard scheduler returns a map of {node: [tasks]}
        wrench_decision = self.scheduler.schedule([task_to_schedule])
        
        for node, tasks in wrench_decision.items():
            if tasks and tasks[0] == task_to_schedule:
                return {node.name: task_to_schedule}
        
        # Fallback if something goes wrong
        return {}