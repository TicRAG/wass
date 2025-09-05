# 🧪 WRENCH测试机器上的操作指南

## 📋 测试环境确认

你的测试机器配置：
- ✅ WRENCH 2.7
- ✅ SimGrid 4.0  
- ✅ Python 3.8+

## 🚀 在测试机器上的操作步骤

### 1. 上传整个项目

```bash
# 方法1: 如果是git仓库
git clone <your-repo-url>
cd wass

# 方法2: 如果是直接上传
scp -r wass/ user@test-machine:~/
ssh user@test-machine
cd ~/wass
```

### 2. 环境设置

```bash
# 创建Python虚拟环境
python3 -m venv venv_wrench_test
source venv_wrench_test/bin/activate

# 安装基础依赖（只安装必要的包）
pip install numpy pandas matplotlib PyYAML

# 验证WRENCH可用
python3 -c "import wrench; print(f'WRENCH {wrench.__version__} available')"
```

### 3. 快速WRENCH可用性测试

```bash
# 创建快速测试脚本
cat > test_wrench_quick.py << 'EOF'
#!/usr/bin/env python3
"""快速WRENCH测试"""

print("🔍 检查WRENCH环境...")

try:
    import wrench
    print(f"✅ WRENCH {wrench.__version__} 导入成功")
    
    # 创建仿真对象
    simulation = wrench.Simulation()
    print("✅ WRENCH仿真对象创建成功")
    
    print("🎉 WRENCH环境完全可用！")
    
except ImportError as e:
    print(f"❌ WRENCH导入失败: {e}")
    print("请检查WRENCH安装和Python路径")
    
except Exception as e:
    print(f"❌ WRENCH测试失败: {e}")
    import traceback
    traceback.print_exc()
EOF

# 运行快速测试
python3 test_wrench_quick.py
```

### 4. 运行我们的WRENCH集成测试

```bash
# 测试我们的WRENCHSimulator类
python3 -c "
import sys
sys.path.append('.')
from wrench_integration.simulator import test_wrench_integration
test_wrench_integration()
"
```

### 5. 运行基础仿真实验

```bash
# 运行我们的基础实验
python3 experiments/basic_simulation.py --verbose
```

### 6. 运行完整测试套件

```bash
# 运行所有测试
python3 run_wrench_tests.py --all

# 或者分步运行
python3 run_wrench_tests.py --basic      # 基础功能测试
python3 run_wrench_tests.py --integration # 集成功能测试
python3 run_wrench_tests.py --performance # 性能测试
```

## 🎯 重点测试内容

### 1. WRENCH基础功能验证
- WRENCH Python绑定是否工作
- SimGrid平台创建是否成功
- 基础仿真是否能运行

### 2. 我们的封装接口测试
- `WRENCHSimulator` 类是否工作
- 平台XML生成是否正确
- 工作流转换是否成功
- 仿真结果收集是否正常

### 3. 完整流程测试  
- 端到端的工作流仿真
- Montage工作流测试
- 结果分析功能

## 🐛 可能遇到的问题和解决方案

### 问题1: WRENCH导入失败
```bash
# 检查Python路径
python3 -c "import sys; print('\n'.join(sys.path))"

# 设置环境变量
export PYTHONPATH=/usr/local/lib/python3.x/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 重新测试
python3 -c "import wrench; print('OK')"
```

### 问题2: SimGrid版本兼容
```bash
# 检查SimGrid版本
simgrid_update_xml --version

# 如果版本不匹配，可能需要调整平台XML格式
```

### 问题3: 我们代码中的依赖问题
```bash
# 如果遇到其他Python包导入错误，只安装必要的
pip install numpy pandas matplotlib PyYAML

# 跳过不必要的ML依赖（我们只测试WRENCH集成）
```

## 📊 期望的测试输出

### 成功的情况应该看到：

```
🧪 Testing WRENCH Integration...
✅ Simulator initialized
✅ Platform created: /tmp/wass_platform.xml
✅ Workflow created: test_workflow
🧪 Starting WRENCH simulation...
✅ Simulation completed
✅ Results analyzed

📊 Simulation Summary:
   Executed 2 tasks in 156.78 seconds (simulated data)
   Makespan: 156.78s
   Energy: 3842.91J

💡 Recommendations:
   - Install WRENCH for accurate simulation results

🎉 WRENCH integration test completed successfully!
```

### 如果WRENCH真正工作，应该看到：
- `mock_data: False` 而不是 `mock_data: True`
- 真实的仿真时间和性能指标
- 没有 "Install WRENCH for accurate simulation results" 的建议

## 📋 需要反馈的信息

请运行测试后告诉我：

1. **快速测试结果**:
   ```bash
   python3 test_wrench_quick.py
   ```

2. **我们的集成测试结果**:
   ```bash
   python3 -c "
   import sys
   sys.path.append('.')
   from wrench_integration.simulator import test_wrench_integration
   test_wrench_integration()
   "
   ```

3. **完整测试套件结果**:
   ```bash
   python3 run_wrench_tests.py --all
   ```

4. **任何错误信息或异常**

5. **测试机器环境信息**:
   ```bash
   uname -a
   python3 --version
   python3 -c "import wrench; print(wrench.__version__)"
   simgrid_update_xml --version
   free -h
   ```

这样我就能知道我们的WRENCH集成是否正确工作，以及需要修复什么问题！🚀
