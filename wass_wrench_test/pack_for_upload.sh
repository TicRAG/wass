#!/bin/bash
# 打包测试包
echo "📦 打包WRENCH测试包..."
cd ..
tar -czf wass_wrench_test.tar.gz wass_wrench_test/
echo "✅ 测试包已创建: wass_wrench_test.tar.gz"
echo "现在可以上传到测试机器了："
echo "scp wass_wrench_test.tar.gz user@test-machine:~/"
