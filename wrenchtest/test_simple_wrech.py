#!/usr/bin/env python3
"""
WRENCH 崩溃定位脚本 (Storage Service 诊断优先)
步骤:
1. 先 STORAGE_ONLY=True 运行: 仅创建平台 + Storage (应不崩)
2. 若成功 -> STORAGE_ONLY=False, CREATE_STORAGE_FIRST=True (先建 storage 再建 compute)
3. 若仍成功 -> 再 SET_ADD_WORKFLOW=True
4. 若 storage 单独就崩 -> 改 ROUTING_FLOYD=True (删除全部 <route>)
5. 若改 mount 点后仍崩 -> 可能是 WRENCH bug (报告最小复现)

切换参数逐步定位在哪一步崩。
"""

import pathlib, random, sys
try:
    import wrench
except ImportError:
    print("需要安装 wrench")
    sys.exit(1)

# ---- 开关 ----
STORAGE_ONLY = False            # 步骤1: True
CREATE_STORAGE_FIRST = True    # 当 STORAGE_ONLY=False 时优先建 storage
SET_ADD_WORKFLOW = True       # 仅在前两步稳定后再设 True
USE_SCRATCH = True            # 暂不使用 scratch
ROUTING_FLOYD = False          # 如 storage 仍崩，再改 True
STORAGE_MOUNT = "/store"       # 改为非根挂载避免特殊处理
# ----------------

PLAT = pathlib.Path("platform_example.xml")
DAX  = pathlib.Path("workflow_example.dax")

def build_platform():
    if ROUTING_FLOYD:
        routing = 'routing="Floyd"'
        routes = ""  # Floyd 不写 <route>
    else:
        routing = 'routing="Full"'
        routes = """
    <route src="ControllerHost" dst="ComputeHost1"><link_ctn id="net"/></route>
    <route src="ControllerHost" dst="ComputeHost2"><link_ctn id="net"/></route>
    <route src="ControllerHost" dst="StorageHost"><link_ctn id="net"/></route>
    <route src="ComputeHost1" dst="ComputeHost2"><link_ctn id="net"/></route>
    <route src="ComputeHost1" dst="StorageHost"><link_ctn id="net"/></route>
    <route src="ComputeHost2" dst="StorageHost"><link_ctn id="net"/></route>
"""
    compute1_disk = ""
    if USE_SCRATCH:
        compute1_disk = f"""
    <disk id="c1d" read_bw="200MBps" write_bw="200MBps">
      <prop id="size" value="80GB"/>
      <prop id="mount" value="/scratch"/>
    </disk>"""
    plat = f"""<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
<platform version="4.1">
  <zone id="z" {routing}>
    <host id="ComputeHost1" speed="2Gf" core="1">{compute1_disk}
    </host>
    <host id="ComputeHost2" speed="2Gf" core="1"/>
    <host id="StorageHost" speed="1Gf" core="1">
      <disk id="sd" read_bw="150MBps" write_bw="150MBps">
        <prop id="size" value="500GB"/>
        <prop id="mount" value="{STORAGE_MOUNT}"/>
      </disk>
    </host>
    <host id="ControllerHost" speed="1Gf" core="1"/>
    <link id="net" bandwidth="1GBps" latency="40us"/>
    {routes}
  </zone>
</platform>"""
    PLAT.write_text(plat, encoding="utf-8")

def build_workflow():
    DAX.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<adag xmlns="https://pegasus.isi.edu/schema/DAX" version="2.1">
  <job id="A" name="Task_A" runtime="5.0" />
  <job id="B" name="Task_B" runtime="3.0" />
  <child ref="B"><parent ref="A"/></child>
</adag>
""", encoding="utf-8")

class SimpleScheduler:
    def __init__(self, sim, hosts, workflow, storage_service, compute_service): 
        self.sim = sim
        self.hosts = hosts
        self.workflow = workflow
        self.storage_service = storage_service
        self.compute_service = compute_service
        self.completed_tasks = set()
        
    def start_scheduling(self):
        """开始调度可运行的任务"""
        self._schedule_ready_tasks()
        
    def handle_event(self, event):
        """处理仿真事件"""
        if event["event_type"] == "standard_job_completion":
            job = event["standard_job"]
            for task in job.get_tasks():
                self.completed_tasks.add(task)
                print(f"[{self.sim.get_simulated_time():.2f}s] 任务完成: {task.get_name()}")
            # Schedule new ready tasks
            self._schedule_ready_tasks()
        elif event["event_type"] == "standard_job_failure":
            job = event["standard_job"]
            print(f"[{self.sim.get_simulated_time():.2f}s] 任务失败: {[t.get_name() for t in job.get_tasks()]}")
        else:
            print(f"调度器收到未处理事件: {event['event_type']}")
            
    def _schedule_ready_tasks(self):
        """调度所有准备就绪的任务"""
        ready_tasks = self.workflow.get_ready_tasks()
        for task in ready_tasks:
            if task not in self.completed_tasks:
                host = self.hosts[0]  # 简单选择第一个主机
                print(f"[{self.sim.get_simulated_time():.2f}s] 调度 {task.get_name()} -> {host}")
                
                # 准备文件位置字典
                file_locations = {}
                for f in task.get_input_files():
                    file_locations[f] = self.storage_service
                for f in task.get_output_files():
                    file_locations[f] = self.storage_service
                
                # 创建标准作业并提交
                job = self.sim.create_standard_job([task], file_locations)
                
                # 提交作业到计算服务
                self.compute_service.submit_standard_job(job)

def main():
    print("诊断运行参数:",
          f"STORAGE_ONLY={STORAGE_ONLY}, CREATE_STORAGE_FIRST={CREATE_STORAGE_FIRST}, "
          f"ADD_WORKFLOW={SET_ADD_WORKFLOW}, ROUTING_FLOYD={ROUTING_FLOYD}, USE_SCRATCH={USE_SCRATCH}")
    build_platform()
    if SET_ADD_WORKFLOW:
        build_workflow()

    sim = wrench.Simulation()
    sim.start(PLAT.read_text(encoding="utf-8"), "ControllerHost")

    storage_service=None
    compute_service=None

    def make_storage():
        print("创建 Storage Service ...")
        return sim.create_simple_storage_service("StorageHost", [STORAGE_MOUNT])

    def make_compute():
        print("创建 Compute Service ...")
        resources={"ComputeHost1":(1,8_589_934_592),"ComputeHost2":(1,8_589_934_592)}
        scratch="" if not USE_SCRATCH else "/scratch"
        return sim.create_bare_metal_compute_service("ComputeHost1", resources, scratch, {}, {})

    if STORAGE_ONLY:
        storage_service = make_storage()
        print("仅 Storage 成功，退出。")
        sim.terminate()
        return

    if CREATE_STORAGE_FIRST:
        storage_service = make_storage()
        compute_service = make_compute()
    else:
        compute_service = make_compute()
        storage_service = make_storage()

    print("基础服务已建成功。")

    if SET_ADD_WORKFLOW:
        print("创建工作流...")
        workflow = sim.create_workflow()
        
        # Add files to the simulation
        input_file = sim.add_file("input.txt", 1024)
        output_file = sim.add_file("output.txt", 1024)
        
        # Create file copy on storage service
        storage_service.create_file_copy(input_file)
        
        # Create tasks based on the DAX content
        task_a = workflow.add_task("Task_A", 5.0 * 1000000000, 1, 1, 0)  # 5 seconds * 1GFlop
        task_b = workflow.add_task("Task_B", 3.0 * 1000000000, 1, 1, 0)  # 3 seconds * 1GFlop
        
        # Add file dependencies - Task B depends on Task A through the output file
        task_a.add_input_file(input_file)
        task_a.add_output_file(output_file)
        task_b.add_input_file(output_file)  # This creates the dependency: B waits for A's output
        
        print("工作流创建完成，启动调度器...")
        scheduler = SimpleScheduler(sim, ["ComputeHost1", "ComputeHost2"], workflow, storage_service, compute_service)
        
        # Start scheduling
        scheduler.start_scheduling()
        
        # Run simulation with workflow completion check
        print("开始仿真循环...")
        while not workflow.is_done():
            evt = sim.wait_for_next_event()
            print(f"[{sim.get_simulated_time():.2f}s] 事件: {evt}")
            
            # Handle different event types
            if evt["event_type"] == "standard_job_completion":
                scheduler.handle_event(evt)
            elif evt["event_type"] == "simulation_termination":
                break
            else:
                print(f"未处理的事件类型: {evt['event_type']}")
                
        print(f"工作流完成 时间={sim.get_simulated_time():.4f}s")
    else:
        print("未启用工作流。")

    sim.terminate()

if __name__=="__main__":
    try:
        main()
    except wrench.WRENCHException as e:
        print("WRENCH 异常:", e)
    finally:
        for f in (PLAT,DAX):
            if f.exists():
                try: f.unlink()
                except: pass
        print("清理完成")