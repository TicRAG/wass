#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试奖励函数修复效果
"""

import sys
import os
import numpy as np
import math

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.drl.reward import compute_final_reward, EpisodeStats
from src.reward_fix import RewardFix

def test_reward_functions():
    """测试奖励函数"""
    print("🧪 测试奖励函数修复效果...")
    
    # 测试原始奖励函数
    print("\n1. 测试原始奖励函数 (compute_final_reward):")
    
    # 测试正常值
    stats_normal = EpisodeStats(makespan=100.0)
    reward_normal = compute_final_reward(stats_normal)
    print(f"   正常makespan (100.0): 奖励={reward_normal:.4f}")
    
    # 测试大值
    stats_large = EpisodeStats(makespan=1e9)
    reward_large = compute_final_reward(stats_large)
    print(f"   大makespan (1e9): 奖励={reward_large:.4f}")
    
    # 测试极大值
    stats_huge = EpisodeStats(makespan=45098466399.18)  # 从错误日志中获取的值
    reward_huge = compute_final_reward(stats_huge)
    print(f"   极大makespan (45098466399.18): 奖励={reward_huge:.4f}")
    
    # 测试带有滚动统计的值
    stats_with_rolling = EpisodeStats(
        makespan=100.0,
        rolling_mean_makespan=120.0,
        rolling_std_makespan=20.0
    )
    reward_with_rolling = compute_final_reward(stats_with_rolling)
    print(f"   带滚动统计的makespan (100.0, mean=120.0, std=20.0): 奖励={reward_with_rolling:.4f}")
    
    # 测试修复后的奖励函数
    print("\n2. 测试修复后的奖励函数 (RewardFix):")
    
    reward_fix = RewardFix()
    
    # 测试正常值
    reward_normal_fix = reward_fix.calculate_normalized_reward(120.0, 100.0, 1.0)
    print(f"   正常值 (teacher=120.0, student=100.0): 奖励={reward_normal_fix:.4f}")
    
    # 测试大值
    reward_large_fix = reward_fix.calculate_normalized_reward(1.2e12, 1.0e12, 1.0)
    print(f"   大值 (teacher=1.2e12, student=1.0e12): 奖励={reward_large_fix:.4f}")
    
    # 测试极大值
    reward_huge_fix = reward_fix.calculate_normalized_reward(5e13, 4.5e13, 1.0)
    print(f"   极大值 (teacher=5e13, student=4.5e13): 奖励={reward_huge_fix:.4f}")
    
    # 测试调试信息
    print("\n3. 测试调试信息输出:")
    reward_fix.debug_reward_info("test_task", 1.2e12, 1.0e12, reward_large_fix)
    
    print("\n✅ 奖励函数测试完成!")

if __name__ == "__main__":
    test_reward_functions()