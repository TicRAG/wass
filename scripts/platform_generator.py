#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WASS-RAG 平台配置生成器
生成不同规模的集群配置文件，用于可扩展性实验
"""

import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import argparse

class PlatformGenerator:
    """平台配置生成器"""
    
    def __init__(self, output_dir: str = "configs/platforms"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_platform_xml(self, 
                           config_name: str,
                           num_compute_nodes: int,
                           node_configs: List[Tuple[int, float]],  # (cores, speed_gflops)
                           network_bandwidth: str = "1.25GBps",
                           network_latency: str = "50us") -> str:
        """创建平台XML配置文件"""
        
        # 创建根元素
        platform = ET.Element("platform", version="4.1")
        
        # 添加区域
        zone = ET.SubElement(platform, "zone", id="world", routing="Full")
        
        # 1. 添加控制节点
        controller = ET.SubElement(zone, "host", id="ControllerHost", speed="10Gf")
        
        # 2. 添加存储节点
        storage = ET.SubElement(zone, "host", id="StorageHost", speed="5Gf")
        storage_disk = ET.SubElement(storage, "disk", id="large_disk", 
                                   read_bw="2GBps", write_bw="1GBps")
        storage_prop = ET.SubElement(storage_disk, "prop", id="size", value="100TB")
        storage_prop2 = ET.SubElement(storage_disk, "prop", id="mount", value="/")
        
        # 3. 添加计算节点
        for i in range(num_compute_nodes):
            node_id = f"ComputeHost{i+1}"
            
            # 循环使用节点配置
            cores, speed_gflops = node_configs[i % len(node_configs)]
            speed_str = f"{speed_gflops}Gf"
            
            host = ET.SubElement(zone, "host", id=node_id, speed=speed_str, core=str(cores))
            
            # 添加本地存储
            disk = ET.SubElement(host, "disk", id=f"disk_{i+1}", 
                               read_bw="800MBps", write_bw="400MBps")
            disk_prop = ET.SubElement(disk, "prop", id="size", value="1TB")
            disk_prop2 = ET.SubElement(disk, "prop", id="mount", value="/tmp")
        
        # 4. 添加网络链接
        # 控制节点到存储节点
        ET.SubElement(zone, "link", id="controller_storage_link",
                     bandwidth=network_bandwidth, latency=network_latency)
        
        # 控制节点到所有计算节点
        for i in range(num_compute_nodes):
            ET.SubElement(zone, "link", id=f"controller_compute{i+1}_link",
                         bandwidth=network_bandwidth, latency=network_latency)
        
        # 存储节点到所有计算节点  
        for i in range(num_compute_nodes):
            ET.SubElement(zone, "link", id=f"storage_compute{i+1}_link",
                         bandwidth=network_bandwidth, latency=network_latency)
        
        # 计算节点之间的连接（网格拓扑）
        for i in range(num_compute_nodes):
            for j in range(i+1, num_compute_nodes):
                ET.SubElement(zone, "link", id=f"compute{i+1}_compute{j+1}_link",
                             bandwidth=network_bandwidth, latency=network_latency)
        
        # 5. 添加路由定义
        route_controller_storage = ET.SubElement(zone, "route", 
                                               src="ControllerHost", dst="StorageHost")
        ET.SubElement(route_controller_storage, "link_ctn", id="controller_storage_link")
        
        for i in range(num_compute_nodes):
            node_id = f"ComputeHost{i+1}"
            
            # 控制节点到计算节点
            route = ET.SubElement(zone, "route", src="ControllerHost", dst=node_id)
            ET.SubElement(route, "link_ctn", id=f"controller_compute{i+1}_link")
            
            # 存储节点到计算节点
            route2 = ET.SubElement(zone, "route", src="StorageHost", dst=node_id)
            ET.SubElement(route2, "link_ctn", id=f"storage_compute{i+1}_link")
        
        # 计算节点之间的路由
        for i in range(num_compute_nodes):
            for j in range(i+1, num_compute_nodes):
                src_id = f"ComputeHost{i+1}"
                dst_id = f"ComputeHost{j+1}"
                
                route = ET.SubElement(zone, "route", src=src_id, dst=dst_id)
                ET.SubElement(route, "link_ctn", id=f"compute{i+1}_compute{j+1}_link")
        
        # 生成格式化的XML
        rough_string = ET.tostring(platform, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # 去除多余的空行
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        formatted_xml = '\n'.join(lines)
        
        # 保存文件
        filename = f"platform_{config_name}.xml"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_xml)
        
        return str(filepath)
    
    def generate_standard_configs(self) -> Dict[str, str]:
        """生成标准配置集合"""
        configs = {}
        
        # 小规模配置 - 16节点
        configs['small'] = self.create_platform_xml(
            config_name='small',
            num_compute_nodes=16,
            node_configs=[
                (4, 2.0),   # 4核，2GHz
                (4, 2.5),   # 4核，2.5GHz  
                (8, 2.0),   # 8核，2GHz
                (8, 3.0),   # 8核，3GHz
            ],
            network_bandwidth="1GBps",
            network_latency="100us"
        )
        
        # 中等规模配置 - 64节点
        configs['medium'] = self.create_platform_xml(
            config_name='medium', 
            num_compute_nodes=64,
            node_configs=[
                (8, 2.5),   # 8核，2.5GHz
                (8, 3.0),   # 8核，3GHz
                (16, 2.0),  # 16核，2GHz
                (16, 2.8),  # 16核，2.8GHz
                (12, 3.2),  # 12核，3.2GHz
            ],
            network_bandwidth="10GBps", 
            network_latency="50us"
        )
        
        # 大规模配置 - 128节点
        configs['large'] = self.create_platform_xml(
            config_name='large',
            num_compute_nodes=128, 
            node_configs=[
                (16, 2.8),  # 16核，2.8GHz
                (16, 3.2),  # 16核，3.2GHz
                (32, 2.5),  # 32核，2.5GHz
                (32, 3.0),  # 32核，3GHz
                (24, 3.5),  # 24核，3.5GHz
                (20, 4.0),  # 20核，4GHz
            ],
            network_bandwidth="25GBps",
            network_latency="20us"
        )
        
        # 超大规模配置 - 256节点（用于极限测试）
        configs['xlarge'] = self.create_platform_xml(
            config_name='xlarge',
            num_compute_nodes=256,
            node_configs=[
                (32, 3.0),  # 32核，3GHz
                (32, 3.5),  # 32核，3.5GHz
                (64, 2.8),  # 64核，2.8GHz
                (64, 3.2),  # 64核，3.2GHz
                (48, 4.0),  # 48核，4GHz
                (40, 4.5),  # 40核，4.5GHz
            ],
            network_bandwidth="100GBps",
            network_latency="10us"
        )
        
        return configs
    
    def create_config_yaml(self, platform_file: str, scale: str) -> str:
        """为平台配置创建对应的YAML配置文件"""
        yaml_content = f"""# WASS-RAG 平台配置 - {scale.upper()}规模
platform:
  platform_file: "{platform_file}"
  controller_host: "ControllerHost"
  storage_host: "StorageHost"
  scale: "{scale}"
  
# 根据规模调整实验参数
experiment:
  scale: "{scale}"
  max_workflow_size: {self._get_max_workflow_size(scale)}
  recommended_episodes: {self._get_recommended_episodes(scale)}
  batch_size: {self._get_batch_size(scale)}

# 计算资源配置
resources:
  memory_limit_gb: {self._get_memory_limit(scale)}
  parallel_jobs: {self._get_parallel_jobs(scale)}
"""
        
        yaml_filename = f"platform_{scale}.yaml"
        yaml_filepath = self.output_dir / yaml_filename
        
        with open(yaml_filepath, 'w', encoding='utf-8') as f:
            f.write(yaml_content)
        
        return str(yaml_filepath)
    
    def _get_max_workflow_size(self, scale: str) -> int:
        """根据规模推荐最大工作流大小"""
        scale_mapping = {
            'small': 200,
            'medium': 1000,
            'large': 3000,
            'xlarge': 5000
        }
        return scale_mapping.get(scale, 500)
    
    def _get_recommended_episodes(self, scale: str) -> int:
        """根据规模推荐训练episode数"""
        scale_mapping = {
            'small': 100,
            'medium': 300,
            'large': 500,
            'xlarge': 1000
        }
        return scale_mapping.get(scale, 200)
    
    def _get_batch_size(self, scale: str) -> int:
        """根据规模推荐批处理大小"""
        scale_mapping = {
            'small': 32,
            'medium': 64,
            'large': 128,
            'xlarge': 256
        }
        return scale_mapping.get(scale, 64)
    
    def _get_memory_limit(self, scale: str) -> int:
        """根据规模推荐内存限制（GB）"""
        scale_mapping = {
            'small': 8,
            'medium': 32,
            'large': 64,
            'xlarge': 128
        }
        return scale_mapping.get(scale, 16)
    
    def _get_parallel_jobs(self, scale: str) -> int:
        """根据规模推荐并行任务数"""
        scale_mapping = {
            'small': 4,
            'medium': 16,
            'large': 32,
            'xlarge': 64
        }
        return scale_mapping.get(scale, 8)
    
    def generate_summary(self, configs: Dict[str, str]) -> str:
        """生成平台配置摘要"""
        summary_path = self.output_dir / "platform_summary.md"
        
        summary_content = f"""# WASS-RAG 平台配置摘要

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 配置概览

| 规模 | 计算节点数 | 网络带宽 | 延迟 | 适用场景 |
|------|------------|----------|------|----------|
| Small | 16 | 1GBps | 100us | 快速测试，算法验证 |
| Medium | 64 | 10GBps | 50us | 常规实验，性能对比 |
| Large | 128 | 25GBps | 20us | 可扩展性测试 |
| XLarge | 256 | 100GBps | 10us | 极限性能测试 |

## 生成的文件

"""
        
        for scale, filepath in configs.items():
            summary_content += f"- `{Path(filepath).name}` - {scale}规模配置\n"
        
        summary_content += """
## 使用说明

1. 根据实验需求选择合适的平台配置
2. 更新 `configs/experiment.yaml` 中的 `platform_file` 字段
3. 运行实验：`python experiments/wrench_real_experiment.py`

## 性能预期

- **Small**: 适合快速迭代和调试
- **Medium**: 标准论文实验规模
- **Large**: 展示可扩展性的关键配置
- **XLarge**: 极限性能测试，需要高性能硬件

"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        return str(summary_path)

def main():
    parser = argparse.ArgumentParser(description='WASS-RAG 平台配置生成器')
    parser.add_argument('--scale', choices=['small', 'medium', 'large', 'xlarge', 'all'],
                       default='all', help='平台规模')
    parser.add_argument('--output', default='configs/platforms',
                       help='输出目录')
    
    args = parser.parse_args()
    
    generator = PlatformGenerator(args.output)
    
    if args.scale == 'all':
        print("🌟 生成完整平台配置集合...")
        configs = generator.generate_standard_configs()
        
        # 为每个平台配置生成对应的YAML文件
        for scale, xml_path in configs.items():
            yaml_path = generator.create_config_yaml(f"configs/platforms/{Path(xml_path).name}", scale)
            print(f"✅ {scale}规模: {Path(xml_path).name} + {Path(yaml_path).name}")
        
        summary_path = generator.generate_summary(configs)
        print(f"\n📋 平台摘要已保存: {summary_path}")
        print(f"🎉 总计生成 {len(configs)} 套平台配置")
        
    else:
        print(f"🚀 生成 {args.scale} 规模平台配置...")
        # 单独生成指定规模的配置
        # 这里需要重构代码以支持单独生成，现在先用all模式
        configs = generator.generate_standard_configs()
        if args.scale in configs:
            xml_path = configs[args.scale]
            yaml_path = generator.create_config_yaml(f"configs/platforms/{Path(xml_path).name}", args.scale)
            print(f"✅ 生成完成: {Path(xml_path).name} + {Path(yaml_path).name}")
        else:
            print(f"❌ 未知规模: {args.scale}")

if __name__ == "__main__":
    main()
