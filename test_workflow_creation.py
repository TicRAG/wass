#!/usr/bin/env python3

import json
import wrench
import requests

# Test workflow data in WfCommons format (similar to the working example)
test_workflow = {
    "name": "test-workflow-5-tasks.json",
    "description": "Test workflow for debugging",
    "createdAt": "2024-01-01T00:00:00.000000+00:00",
    "schemaVersion": "1.5",
    "author": {
        "name": "test",
        "email": "test@example.com"
    },
    "workflow": {
        "specification": {
            "tasks": [
                {
                    "name": "task_0",
                    "id": "task_0",
                    "children": ["task_1", "task_2"],
                    "inputFiles": ["input_0.txt"],
                    "outputFiles": ["output_0.txt"],
                    "parents": []
                },
                {
                    "name": "task_1", 
                    "id": "task_1",
                    "children": ["task_3"],
                    "inputFiles": ["output_0.txt"],
                    "outputFiles": ["output_1.txt"],
                    "parents": ["task_0"]
                },
                {
                    "name": "task_2",
                    "id": "task_2", 
                    "children": ["task_3"],
                    "inputFiles": ["output_0.txt"],
                    "outputFiles": ["output_2.txt"],
                    "parents": ["task_0"]
                },
                {
                    "name": "task_3",
                    "id": "task_3",
                    "children": ["task_4"],
                    "inputFiles": ["output_1.txt", "output_2.txt"],
                    "outputFiles": ["output_3.txt"],
                    "parents": ["task_1", "task_2"]
                },
                {
                    "name": "task_4",
                    "id": "task_4",
                    "children": [],
                    "inputFiles": ["output_3.txt"],
                    "outputFiles": ["final_output.txt"],
                    "parents": ["task_3"]
                }
            ],
            "files": [
                {"name": "input_0.txt", "size": 1000},
                {"name": "output_0.txt", "size": 2000},
                {"name": "output_1.txt", "size": 1500},
                {"name": "output_2.txt", "size": 1500},
                {"name": "output_3.txt", "size": 3000},
                {"name": "final_output.txt", "size": 5000}
            ]
        }
    },
    "execution": {
        "workflow_executions": [{
            "name": "test-execution",
            "tasks": [
                {"name": "task_0", "runtime": 10.0, "cores": 1, "flops": 1000000},
                {"name": "task_1", "runtime": 15.0, "cores": 1, "flops": 1500000},
                {"name": "task_2", "runtime": 12.0, "cores": 1, "flops": 1200000},
                {"name": "task_3", "runtime": 20.0, "cores": 1, "flops": 2000000},
                {"name": "task_4", "runtime": 8.0, "cores": 1, "flops": 800000}
            ]
        }]
    }
}

print("Testing workflow creation...")

# Test 1: Direct simulation.create_workflow_from_json
try:
    print("\n=== Test 1: Direct simulation.create_workflow_from_json ===")
    simulation = wrench.Simulation()
    
    # Start simulation with minimal platform
    platform_xml = """<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="AS0" routing="Full">
    <host id="UserHost" speed="1Gf" core="1">
      <disk id="disk" read_bw="100MBps" write_bw="100MBps">
        <prop id="size" value="100GB"/>
        <prop id="mount" value="/"/>
      </disk>
    </host>
  </zone>
</platform>"""
    
    simulation.start(platform_xml, "UserHost")
    
    # Try creating workflow directly
    workflow = simulation.create_workflow_from_json(
        test_workflow,
        reference_flop_rate="100Mf",
        ignore_machine_specs=True,
        redundant_dependencies=False,
        ignore_cycle_creating_dependencies=False,
        min_cores_per_task=1,
        max_cores_per_task=1,
        enforce_num_cores=True,
        ignore_avg_cpu=True,
        show_warnings=True
    )
    
    print(f"Workflow created successfully: {workflow.get_name()}")
    print(f"Number of tasks: {len(workflow.get_tasks())}")
    print(f"Number of input files: {len(workflow.get_input_files())}")
    
    # Test getting ready tasks
    ready_tasks = workflow.get_ready_tasks()
    print(f"Ready tasks: {len(ready_tasks)}")
    for task in ready_tasks:
        print(f"  - {task.get_name()}")
    
except Exception as e:
    print(f"Direct creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Daemon API approach
try:
    print("\n=== Test 2: Daemon API approach ===")
    
    # Add workflow_name field for daemon
    test_workflow["workflow_name"] = "test-workflow-5-tasks"
    
    # Try daemon API
    response = requests.post(
        "http://localhost:8080/createWorkflowFromJSON",
        json=test_workflow,
        params={
            "reference_flop_rate": "100Mf",
            "ignore_machine_specs": "true",
            "redundant_dependencies": "false", 
            "ignore_cycle_creating_dependencies": "false",
            "min_cores_per_task": "1",
            "max_cores_per_task": "1",
            "enforce_num_cores": "true",
            "ignore_avg_cpu": "true",
            "show_warnings": "true"
        }
    )
    
    print(f"Daemon response status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Daemon response keys: {list(result.keys())}")
        if 'workflow_name' in result:
            workflow_name = result['workflow_name']
            print(f"Workflow name from daemon: {workflow_name}")
            
            # Create workflow object manually
            workflow = wrench.Workflow(simulation, workflow_name)
            print(f"Manual workflow creation successful: {workflow.get_name()}")
            print(f"Number of tasks: {len(workflow.get_tasks())}")
            
            # Test getting ready tasks
            ready_tasks = workflow.get_ready_tasks()
            print(f"Ready tasks: {len(ready_tasks)}")
            for task in ready_tasks:
                print(f"  - {task.get_name()}")
    else:
        print(f"Daemon error: {response.text}")
        
except Exception as e:
    print(f"Daemon API approach failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test completed ===")