# WRENCH测试包使用说明

## 📦 测试包内容

这个测试包包含了所有必要的WASS-RAG WRENCH集成测试代码，可以直接在有WRENCH环境的测试机器上运行。

## 🎯 测试环境要求

- **WRENCH**: 2.7 (已安装)
- **SimGrid**: 4.0 (已安装)  
- **Python**: 3.8+
- **内存**: 8GB+ 推荐
- **磁盘**: 2GB+ 可用空间

## 🚀 快速开始

### 1. 准备测试包

在开发机器上打包文件：

```bash
# 创建测试包目录
mkdir wass_wrench_test

# 拷贝必要文件
cp -r wrench_integration/ wass_wrench_test/
cp -r experiments/ wass_wrench_test/
cp -r src/ wass_wrench_test/
cp -r configs/ wass_wrench_test/
cp requirements_wrench.txt wass_wrench_test/
cp run_wrench_tests.py wass_wrench_test/
cp WRENCH_TEST_GUIDE.md wass_wrench_test/
cp README_WRENCH_TEST.md wass_wrench_test/

# 创建__init__.py文件
touch wass_wrench_test/wrench_integration/__init__.py
touch wass_wrench_test/experiments/__init__.py
touch wass_wrench_test/src/__init__.py

# 打包
tar -czf wass_wrench_test.tar.gz wass_wrench_test/
```

### 2. 上传到测试机器

```bash
# 上传测试包
scp wass_wrench_test.tar.gz user@test-machine:~/

# 登录测试机器
ssh user@test-machine
```

### 3. 在测试机器上解压和设置

```bash
# 解压测试包
cd ~/
tar -xzf wass_wrench_test.tar.gz
cd wass_wrench_test/

# 创建Python虚拟环境
python3 -m venv venv_wrench_test
source venv_wrench_test/bin/activate

# 安装依赖
pip install -r requirements_wrench.txt

# 验证WRENCH可用
python -c "import wrench; print(f'WRENCH {wrench.__version__} available')"
```

### 4. 运行测试

```bash
# 运行所有测试
python run_wrench_tests.py --all

# 或者分步运行
python run_wrench_tests.py --basic      # 基础功能测试
python run_wrench_tests.py --integration # 集成功能测试  
python run_wrench_tests.py --performance # 性能测试
```

## 🧪 测试内容

### 基础测试 (--basic)
1. **WRENCH模块导入** - 验证WRENCH Python绑定
2. **平台创建** - 测试SimGrid平台XML生成
3. **工作流创建** - 测试WRENCH工作流对象创建

### 集成测试 (--integration)  
1. **基础仿真实验** - 运行完整的工作流仿真
2. **WRENCH直接接口** - 测试WRENCH原生API调用

### 性能测试 (--performance)
1. **大规模工作流** - 测试100任务工作流仿真
2. **内存使用** - 监控内存消耗
3. **执行时间** - 测量仿真性能

## 📊 期待的测试结果

### 成功的测试应该显示：

```
🚀 WASS-RAG WRENCH测试开始
============================================================
🔍 检查WRENCH环境...
✅ WRENCH 2.7.x 可用

==================================================
🧪 运行基础测试
==================================================

📋 测试 1: WRENCH模块基础功能
🧪 Testing WRENCH Integration...
✅ Simulator initialized
✅ Platform created: /tmp/wass_platform.xml
✅ Workflow created: test_workflow
✅ Simulation completed
✅ Results analyzed
...
🎉 WRENCH integration test completed successfully!
✅ WRENCH模块测试通过

📋 测试 2: 平台创建功能
✅ 平台文件创建成功: /tmp/wass_platform.xml

📋 测试 3: 工作流创建功能
✅ 工作流创建成功: test_workflow

📊 基础测试结果: 3/3 通过
...

============================================================
📋 测试总结
============================================================
总体结果: X/Y 通过 (Z%)
基础测试: 3/3 通过 (100%)
集成测试: X/Y 通过 (Z%)
性能测试: X/Y 通过 (Z%)

🎉 测试结果良好！
```

## 🐛 常见问题和解决

### 1. WRENCH导入失败

```bash
# 检查WRENCH安装
which python3
python3 -c "import sys; print(sys.path)"

# 设置环境变量
export PYTHONPATH=/usr/local/lib/python3.x/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

### 2. SimGrid平台文件错误

```bash
# 检查SimGrid版本
simgrid_update_xml --version

# 验证平台文件
simgrid_update_xml test_platform.xml
```

### 3. 内存不足

```bash
# 监控内存使用
free -h
htop

# 减少测试规模
# 编辑run_wrench_tests.py，减少大工作流的任务数量
```

### 4. 权限问题

```bash
# 确保有写入权限
chmod +w .
ls -la
```

## 📋 测试报告

测试完成后会生成：

1. **test_report_YYYYMMDD_HHMMSS.json** - 详细的JSON格式测试报告
2. **test_results_YYYYMMDD_HHMMSS/** - 仿真结果目录
3. **控制台输出** - 实时测试进度和结果

### 请提供给我们：

1. **完整的控制台输出** (复制粘贴或截图)
2. **生成的test_report_*.json文件**
3. **任何错误信息或异常堆栈跟踪**
4. **测试机器的环境信息**:
   - 操作系统版本 (`uname -a`)
   - WRENCH版本 (`python -c "import wrench; print(wrench.__version__)"`)
   - SimGrid版本 (`simgrid_update_xml --version`)
   - Python版本 (`python3 --version`)
   - 可用内存 (`free -h`)

## 🔄 下一步

根据测试结果，我们将：

1. **修复发现的问题** - 更新代码解决bug
2. **优化性能** - 改进仿真效率
3. **扩展功能** - 添加更多WRENCH特性
4. **完善文档** - 更新使用指南

---

**联系**: 测试完成后请及时反馈结果，这样我们可以快速迭代改进！
