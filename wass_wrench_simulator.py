import heapq
import logging
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import wrench.workflows as w
from wrench.events import Event, EventType
from wrench.simulation import Simulation
from wrench.utils import get_logger

from src.ai_schedulers import WASSScheduler
from src.utils import get_running_tasks_from_action_map

logger = get_logger(__name__, "INFO")


class WassWrenchSimulator(Simulation):
    """A simulator for WASS that uses a discrete-event simulation approach."""

    def __init__(self, scheduler: WASSScheduler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.event_queue = []
        self.completed_tasks: Set[w.Task] = set()
        self.task_to_node_map: Dict[w.Task, str] = {}
        self.node_earliest_finish_time: Dict[str, float] = defaultdict(float)
        self.submitted_tasks: Set[w.Task] = set()
        self.task_completion_times: Dict[w.Task, float] = {}
        self.total_cpu_time_used = 0.0

    def _reset_state(self):
        """Resets the simulation state for a new run."""
        self.event_queue = []
        self.completed_tasks = set()
        self.task_to_node_map = {}
        self.node_earliest_finish_time = defaultdict(float)
        self.submitted_tasks = set()
        self.task_completion_times = {}
        self.total_cpu_time_used = 0.0

    def _initialize_simulation(self):
        """Initializes the simulation with the entry tasks."""
        initial_ready_tasks = self.workflow.get_ready_tasks()
        for task in initial_ready_tasks:
            heapq.heappush(self.event_queue, Event(0.0, EventType.TASK_READY, task))
            self.submitted_tasks.add(task)

    def run(self) -> Dict[str, float]:
        """Runs the simulation and returns performance metrics."""
        self._reset_state()
        self._initialize_simulation()

        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            current_time = event.time
            event_type = event.event_type
            event_data = event.event_data

            if event_type == EventType.TASK_FINISH:
                self._handle_task_finish(current_time, event_data)
            elif event_type == EventType.TASK_READY:
                self._handle_task_ready(current_time, event_data)

        makespan = self._calculate_makespan()
        avg_cpu_util = self._calculate_avg_cpu_util(makespan)
        data_locality = self._calculate_data_locality()

        return {
            "makespan": makespan,
            "avg_cpu_util": avg_cpu_util,
            "data_locality": data_locality,
        }

    def _handle_task_finish(self, current_time: float, task: w.Task):
        """Handles the completion of a task."""
        logger.debug(f"Time {current_time:.2f}: Task {task.id} finished.")
        self.completed_tasks.add(task)
        self.task_completion_times[task] = current_time

        # Update total CPU time used
        self.total_cpu_time_used += task.computation_size

        for child in task.children:
            if child not in self.submitted_tasks and all(
                p in self.completed_tasks for p in child.parents
            ):
                heapq.heappush(
                    self.event_queue, Event(current_time, EventType.TASK_READY, child)
                )
                self.submitted_tasks.add(child)

    def _handle_task_ready(self, current_time: float, ready_task: w.Task):
        """Handles a task becoming ready for scheduling."""
        logger.debug(f"Time {current_time:.2f}: Task {ready_task.id} is ready.")
        # The scheduler needs to decide where to place this single task
        scheduling_decision = self.scheduler.schedule([ready_task], self)
        
        if not scheduling_decision:
             logger.warning(f"Scheduler returned no decision for task {ready_task.id}")
             return

        node_id = list(scheduling_decision.keys())[0]
        task = list(scheduling_decision.values())[0]
        
        self.task_to_node_map[task] = node_id

        # Calculate start time, considering data transfer and node availability
        est_start_time, data_transfer_time = self._calculate_estimated_start_time(
            task, node_id, current_time
        )
        
        # execution_time = self.platform.get_compute_time(
        #     task.computation_size, self.platform.get_node(node_id).speed
        # )

        execution_time = task.computation_size / self.platform.get_node(node_id).speed
        
        finish_time = est_start_time + execution_time

        # Update node's earliest finish time
        self.node_earliest_finish_time[node_id] = finish_time

        heapq.heappush(
            self.event_queue, Event(finish_time, EventType.TASK_FINISH, task)
        )
        logger.debug(
            f"Time {current_time:.2f}: Scheduled {task.id} on {node_id}. "
            f"EST: {est_start_time:.2f}, Finish: {finish_time:.2f}"
        )

    def _calculate_estimated_start_time(
        self, task: w.Task, node_id: str, current_time: float
    ) -> Tuple[float, float]:
        """Calculates the estimated start time for a task on a given node."""
        node_available_time = self.node_earliest_finish_time.get(node_id, 0.0)

        # Find the time when all parent data is available at the target node
        data_ready_time = 0.0
        total_data_transfer_time = 0.0

        for parent in task.parents:
            parent_node_id = self.task_to_node_map[parent]
            parent_finish_time = self.task_completion_times[parent]

            data_transfer_time = 0.0
            if parent_node_id != node_id:
                # Get data size from workflow edge
                data_size = self.workflow.get_edge_data_size(parent.id, task.id)
                # data_transfer_time = self.platform.get_communication_time(
                #     data_size, parent_node_id, node_id
                # )
                bandwidth = self.platform.get_bandwidth(parent_node_id, node_id)
                data_transfer_time = data_size / bandwidth


            total_data_transfer_time += data_transfer_time
            data_ready_time = max(data_ready_time, parent_finish_time + data_transfer_time)

        # The task can only start when the node is free AND all data has arrived.
        # It also cannot start before it is officially "ready" (current_time).
        start_time = max(node_available_time, data_ready_time, current_time)
        return start_time, total_data_transfer_time

    def _calculate_makespan(self) -> float:
        """Calculates the makespan of the workflow."""
        if not self.task_completion_times:
            return 0.0
        return max(self.task_completion_times.values())

    def _calculate_avg_cpu_util(self, makespan: float) -> float:
        """Calculates the average CPU utilization."""
        if makespan == 0.0:
            return 0.0
        total_possible_cpu_time = sum(
            n.speed for n in self.platform.get_nodes().values()
        ) * makespan
        if total_possible_cpu_time == 0:
            return 0.0
        return self.total_cpu_time_used / total_possible_cpu_time

    def _calculate_data_locality(self) -> float:
        """Calculates the data locality."""
        local_transfers = 0
        total_transfers = 0
        for task in self.workflow.tasks:
            if not task.parents:
                continue
            
            # This check is important because some tasks might not have been scheduled
            # in a failed or incomplete simulation.
            if task not in self.task_to_node_map:
                continue

            child_node = self.task_to_node_map[task]
            for parent in task.parents:
                if parent not in self.task_to_node_map:
                    continue
                
                parent_node = self.task_to_node_map[parent]
                total_transfers += 1
                if parent_node == child_node:
                    local_transfers += 1

        if total_transfers == 0:
            return 1.0  # Or 0.0, depending on definition. 1.0 implies perfect locality.
        return local_transfers / total_transfers