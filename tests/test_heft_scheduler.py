import pytest
from src.wrench_schedulers import HEFTScheduler
from unittest.mock import Mock

# 模拟3个任务的工作流
def create_micro_workflow():
    class Task:
            def __init__(self, name, flops, parents=None):
                self.name = name
                self._flops = flops
                self.parents = parents or []
                self.state = 'NOT_SUBMITTED'
    
            def mark_completed(self):
                self.state = 'COMPLETED'
                
            def get_name(self):
                return self.name
                
            def get_children(self):
                # 在这个简单的测试中，我们通过遍历所有任务来查找子任务
                # 这不是最优实现，但对于测试足够了
                return []
                
            def get_flops(self):
                return self._flops
                
            def get_id(self):
                return self.name
                
            def get_state_as_string(self):
                return self.state
    
    class Workflow:
        def __init__(self, tasks):
            self._tasks = tasks
        def get_runnable_tasks(self):
            return [t for t in self._tasks if all(p.state == 'COMPLETED' for p in t.parents)]
        def get_tasks(self):
            return self._tasks

    # 任务依赖：t3 -> t2 -> t1（修正依赖方向）
    t3 = Task('t3', 20)
    t2 = Task('t2', 15, parents=[])
    t1 = Task('t1', 10, parents=[t2])
    t2.parents.append(t3)  # 显式建立t3->t2依赖

    return Workflow([t1, t2, t3])

# 测试HEFT调度器核心逻辑
def test_heft_scheduler():
    # 创建模拟计算服务
    mock_cs = Mock()
    mock_cs.get_host_current_makespan.side_effect = lambda h: 0.0
    
    # 配置两个不同速度的主机
    hosts = {
        'fast_host': (4, 20.0),  # 4核心，20 GFlop/s
        'slow_host': (2, 10.0)   # 2核心，10 GFlop/s
    }
    
    # 创建模拟仿真器
    mock_sim = Mock()
    
    # 配置Mock以返回正确的任务名称
    job_tasks = {}  # 用于存储job到task的映射
    
    def create_standard_job_side_effect(task_list, *args, **kwargs):
        job_mock = Mock()
        task = task_list[0]  # 获取任务列表中的第一个任务
        job_mock.get_name.return_value = task.get_name()
        job_tasks[job_mock] = task  # 存储映射
        return job_mock
    
    mock_sim.create_standard_job.side_effect = create_standard_job_side_effect
    
    # 配置submit_standard_job以存储调用信息
    submitted_jobs = []
    
    def submit_standard_job_side_effect(job, *args, **kwargs):
        submitted_jobs.append((job, kwargs))
        # 标记任务为完成，以便下一个任务可以运行
        task = job_tasks[job]
        task.mark_completed()
        return Mock()
    
    mock_cs.submit_standard_job.side_effect = submit_standard_job_side_effect
    
    # 初始化调度器
    scheduler = HEFTScheduler(
        simulation=mock_sim,
        compute_service=mock_cs,
        hosts=hosts
    )
    
    # 运行调度
    workflow = create_micro_workflow()
    
    # 多次调用submit_ready_tasks直到所有任务都被提交
    while len(submitted_jobs) < 3:
        scheduler.submit_ready_tasks(workflow)
    
    # 验证调度顺序应为t3 -> t2 -> t1
    submit_order = [job_tasks[job].get_name() for job, _ in submitted_jobs]
    assert submit_order == ['t3', 't2', 't1'], "任务调度顺序不符合HEFT优先级"
    
    # 验证主机分配（t3应分配到最快主机）
    first_job_host = submitted_jobs[0][1]['target_host']
    assert first_job_host == 'fast_host', "首个任务未分配到最快主机"
    
    # 验证完成时间计算
    tasks = workflow.get_tasks()
    t3_flops = 20
    expected_t3_time = t3_flops / 20.0  # 在fast_host执行时间
    assert scheduler.get_earliest_finish_time(tasks[2], 'fast_host') == expected_t3_time

# 验证极端场景
# 在test_heft_with_single_host测试中配置Mock返回值
def test_heft_with_single_host():
    mock_cs = Mock()
    mock_cs.get_host_current_makespan.return_value = 0.0  # 添加返回值配置
    
    # 创建模拟仿真器
    mock_sim = Mock()
    
    # 配置Mock以返回正确的任务名称
    job_tasks = {}  # 用于存储job到task的映射
    
    def create_standard_job_side_effect(task_list, *args, **kwargs):
        job_mock = Mock()
        task = task_list[0]  # 获取任务列表中的第一个任务
        job_mock.get_name.return_value = task.get_name()
        job_tasks[job_mock] = task  # 存储映射
        return job_mock
    
    mock_sim.create_standard_job.side_effect = create_standard_job_side_effect
    
    # 配置submit_standard_job以存储调用信息
    submitted_jobs = []
    
    def submit_standard_job_side_effect(job, *args, **kwargs):
        submitted_jobs.append((job, kwargs))
        # 标记任务为完成，以便下一个任务可以运行
        task = job_tasks[job]
        task.mark_completed()
        return Mock()
    
    mock_cs.submit_standard_job.side_effect = submit_standard_job_side_effect
    
    hosts = {'single_host': (1, 5.0)}

    scheduler = HEFTScheduler(mock_sim, mock_cs, hosts)
    workflow = create_micro_workflow()
    
    # 多次调用submit_ready_tasks直到所有任务都被提交
    while len(submitted_jobs) < 3:
        scheduler.submit_ready_tasks(workflow)
    
    # 获取提交顺序
    submit_order = [job_tasks[job].get_name() for job, _ in submitted_jobs]

    # 所有任务都应分配到唯一主机
    for _, kwargs in submitted_jobs:
        assert kwargs['target_host'] == 'single_host'

    # 添加调试日志输出
    try:
        assert submit_order == ['t3', 't2', 't1'], f"实际调度顺序: {submit_order}"
    except AssertionError as e:
        print(f"[DEBUG] 任务提交顺序: {submit_order}")
        raise


# 在submit_job回调中自动标记任务完成
submitted_tasks = []
def _submit_job(job, target_host):
    task = job.get_tasks()[0]
    task.mark_completed()
    submitted_tasks.append(task)