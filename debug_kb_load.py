#!/usr/bin/env python3
"""调试知识库加载问题"""

import json
import sys
from pathlib import Path

def debug_json_file(filename):
    """调试JSON文件"""
    print(f"调试文件: {filename}")
    
    # 检查文件是否存在
    path = Path(filename)
    if not path.exists():
        print(f"❌ 文件不存在: {filename}")
        return False
    
    # 检查文件大小
    file_size = path.stat().st_size
    print(f"文件大小: {file_size} 字节")
    
    if file_size == 0:
        print(f"❌ 文件为空")
        return False
    
    try:
        # 读取文件内容
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"文件内容长度: {len(content)} 字符")
        print(f"前100个字符: {content[:100]}")
        print(f"后100个字符: {content[-100:]}")
        
        # 尝试解析JSON
        data = json.loads(content)
        print(f"✅ JSON解析成功")
        print(f"数据类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"顶层键: {list(data.keys())}")
            if 'cases' in data:
                print(f"案例数量: {len(data['cases'])}")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        print(f"错误位置: 行 {e.lineno}, 列 {e.colno}")
        
        # 显示错误附近的字符
        if hasattr(e, 'pos'):
            start = max(0, e.pos - 50)
            end = min(len(content), e.pos + 50)
            print(f"错误附近的内容: {content[start:end]}")
        
        return False
    except Exception as e:
        print(f"❌ 其他错误: {e}")
        return False

if __name__ == "__main__":
    # 测试知识库文件
    kb_files = [
        "data/real_heuristic_kb.json",
        "data/wrench_rag_knowledge_base.json"
    ]
    
    for kb_file in kb_files:
        print("=" * 60)
        success = debug_json_file(kb_file)
        print(f"结果: {'✅ 成功' if success else '❌ 失败'}")
        print()