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

    # 在Mock计算服务中返回数值类型
    mock_cs.get_host_current_makespan.return_value = 0.0
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
    
    # 初始化调度器
    scheduler = HEFTScheduler(
        simulation=Mock(),
        compute_service=mock_cs,
        hosts=hosts
    )
    
    # 运行调度
    workflow = create_micro_workflow()
    scheduler.submit_ready_tasks(workflow)
    
    # 验证调度顺序应为t3 -> t2 -> t1
    submit_order = [call.args[0].get_name() for call in mock_cs.submit_standard_job.call_args_list]
    assert submit_order == ['t3', 't2', 't1'], "任务调度顺序不符合HEFT优先级"
    
    # 验证主机分配（t3应分配到最快主机）
    first_job_host = mock_cs.submit_standard_job.call_args_list[0][1]['target_host']
    assert first_job_host == 'fast_host', "首个任务未分配到最快主机"
    
    # 验证完成时间计算
    t3_flops = 20
    expected_t3_time = t3_flops / 20.0  # 在fast_host执行时间
    assert scheduler.get_earliest_finish_time(workflow[2], 'fast_host') == expected_t3_time

# 验证极端场景
# 在test_heft_with_single_host测试中配置Mock返回值
def test_heft_with_single_host():
    mock_cs = Mock()
    mock_cs.get_host_current_makespan.return_value = 0.0  # 添加返回值配置
    hosts = {'single_host': (1, 5.0)}
    
    scheduler = HEFTScheduler(Mock(), mock_cs, hosts)
    workflow = create_micro_workflow()
    scheduler.submit_ready_tasks(workflow)
    
    # 所有任务都应分配到唯一主机
    for call in mock_cs.submit_standard_job.call_args_list:
        assert call[1]['target_host'] == 'single_host'
    
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
mock_cs.submit_standard_job.side_effect = _submit_job

# 分三个阶段执行调度
for _ in range(3):
    # 增强任务状态追踪
    print(f'[PRE] 就绪任务: {[t.name for t in workflow.get_runnable_tasks()]}')
    for i in range(3):
        print(f'--- 调度轮次 {i+1} ---')
        scheduler.submit_ready_tasks(workflow)
        print(f'提交任务: {[t.name for t in submitted_tasks[-1:]]}' if submitted_tasks else '无任务提交')
        print(f'当前主机可用时间: {scheduler.cs.get_host_current_makespan.mock_calls[-1].args if scheduler.cs.method_calls else "未更新"}')
    
    assert [t.name for t in submitted_tasks] == ['t3', 't2', 't1'], f"实际调度顺序: {[t.name for t in submitted_tasks]}"