# WRENCH测试指南

## 🎯 测试环境要求

**测试机器环境**:
- WRENCH 2.7
- SimGrid 4.0
- Python 3.8+

## 📦 测试包准备

### 1. 需要拷贝的文件和目录

```bash
# 核心文件
wass/
├── wrench_integration/          # WRENCH集成模块
│   ├── __init__.py
│   ├── simulator.py            # 主要测试文件
│   ├── platform_builder.py
│   ├── workflow_converter.py
│   └── results_collector.py
├── experiments/                 # 实验脚本
│   ├── __init__.py
│   └── basic_simulation.py     # 基础测试实验
├── src/                        # 基础架构
│   ├── __init__.py
│   ├── config_loader.py
│   ├── interfaces.py
│   └── factory.py
├── tests/                      # 测试脚本
│   ├── __init__.py
│   ├── test_wrench_integration.py
│   └── test_basic_simulation.py
├── configs/                    # 配置文件
│   ├── experiment.yaml
│   └── platform_test.yaml
├── requirements_wrench.txt     # WRENCH测试依赖
├── run_wrench_tests.py        # 测试运行脚本
└── README_WRENCH_TEST.md      # 测试说明
```

### 2. 在测试机器上的操作步骤

```bash
# 1. 上传测试包
scp -r wass_wrench_test/ user@test-machine:~/

# 2. 登录测试机器
ssh user@test-machine

# 3. 进入测试目录
cd ~/wass_wrench_test/

# 4. 设置Python环境
python3 -m venv venv_wrench_test
source venv_wrench_test/bin/activate

# 5. 安装依赖
pip install -r requirements_wrench.txt

# 6. 验证WRENCH
python -c "import wrench; print(f'WRENCH {wrench.__version__} available')"

# 7. 运行测试
python run_wrench_tests.py --all
```

## 🧪 测试计划

### Phase 1: 基础验证测试
1. **WRENCH导入测试** - 验证WRENCH Python绑定
2. **平台创建测试** - 测试SimGrid平台XML生成
3. **工作流创建测试** - 测试WRENCH工作流对象创建
4. **基础仿真测试** - 运行简单的仿真

### Phase 2: 集成功能测试
1. **WASS-WRENCH接口测试** - 测试我们的封装接口
2. **Montage工作流测试** - 测试真实工作流仿真
3. **多平台配置测试** - 测试不同的计算平台
4. **结果收集测试** - 验证仿真结果收集

### Phase 3: 性能和稳定性测试
1. **大规模工作流测试** - 测试复杂工作流
2. **长时间运行测试** - 稳定性验证
3. **内存使用测试** - 资源消耗分析
4. **错误处理测试** - 异常情况处理

## 📋 测试检查清单

### 环境检查
- [ ] WRENCH 2.7 正确安装
- [ ] SimGrid 4.0 正确安装  
- [ ] Python 3.8+ 可用
- [ ] 足够的内存 (建议8GB+)
- [ ] 足够的磁盘空间 (2GB+)

### 功能测试
- [ ] WRENCH Python导入成功
- [ ] 平台XML生成正确
- [ ] 工作流对象创建成功
- [ ] 基础仿真运行成功
- [ ] 结果数据收集正确

### 集成测试
- [ ] WRENCHSimulator类初始化
- [ ] create_platform()方法工作
- [ ] create_workflow()方法工作
- [ ] run_simulation()方法工作
- [ ] analyze_results()方法工作

### 错误处理测试
- [ ] 无效平台配置处理
- [ ] 无效工作流规格处理
- [ ] 仿真超时处理
- [ ] 内存不足处理

## 🐛 常见问题和解决方案

### 1. WRENCH导入失败
```bash
# 检查WRENCH安装
python -c "import sys; print(sys.path)"
export PYTHONPATH=/usr/local/lib/python3.x/site-packages:$PYTHONPATH
```

### 2. SimGrid版本兼容性
```bash
# 检查SimGrid版本
simgrid_update_xml --version
# 确保使用正确的平台XML格式
```

### 3. 内存不足错误
```bash
# 增加交换空间或减少工作流规模
export SIMGRID_CONTEXT_STACK_SIZE=8192
```

### 4. 编译错误
```bash
# 确保编译环境正确
export CXX=g++
export CC=gcc
```

## 📊 测试报告格式

测试完成后，请提供以下信息：

### 环境信息
- 操作系统版本
- WRENCH版本
- SimGrid版本  
- Python版本
- 硬件配置

### 测试结果
- 通过的测试数量
- 失败的测试及错误信息
- 性能指标（仿真时间、内存使用等）
- 生成的结果文件

### 问题反馈
- 遇到的问题及解决方案
- 建议的改进点
- 需要修复的bug

## 🚀 下一步计划

根据测试结果，我们将：

1. **修复发现的问题** - 根据测试反馈修复bug
2. **优化性能** - 改进仿真性能和内存使用
3. **扩展功能** - 添加更多WRENCH特性支持
4. **完善文档** - 更新安装和使用指南

---

**联系方式**: 如有问题请及时反馈测试结果和遇到的问题。
