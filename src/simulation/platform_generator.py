# src/simulation/platform_generator.py
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

def create_platform(scale: str, output_file: str):
    """
    根据规模动态生成一个WRENCH/SimGrid平台XML文件。
    """
    
    # 定义3种规模的规格
    # (CPU 节点数, GPU 节点数)
    specs = {
        "small": {"cpu": 12, "gpu": 4},   # 总共 16 节点
        "medium": {"cpu": 48, "gpu": 16},  # 总共 64 节点
        "large": {"cpu": 96, "gpu": 32}   # 总共 128 节点
    }

    # 定义节点类型
    # (Speed, Cores, RAM) - 这里我们修复了内存问题
    host_types = {
        "cpu": ("100Gf", "16", "64GB"),
        "gpu": ("2000Gf", "32", "256GB")  # 2000Gf 代表一个高速GPU节点
    }

    if scale not in specs:
        raise ValueError(f"未知的规模: {scale}. 可选项: {list(specs.keys())}")

    print(f"--- 正在生成 '{scale}' 规模平台 ({output_file}) ---")
    
    # 1. 创建根元素
    platform = ET.Element("platform", version="4.1")
    zone = ET.SubElement(platform, "zone", id="main", routing="Full")

    # 2. 添加固定主机 (Controller, Storage)
    ET.SubElement(zone, "host", id="ControllerHost", speed="1Gf", core="1")
    storage_host = ET.SubElement(zone, "host", id="StorageHost", speed="1Gf", core="1")
    disk = ET.SubElement(storage_host, "disk", id="storage_disk", read_bw="150MBps", write_bw="150MBps")
    ET.SubElement(disk, "prop", id="size", value="1000GB")
    ET.SubElement(disk, "prop", id="mount", value="/storage")

    # 3. 动态添加计算主机
    compute_hosts = []
    num_cpu = specs[scale]["cpu"]
    num_gpu = specs[scale]["gpu"]

    for i in range(num_cpu):
        host_id = f"cpu_host_{i}"
        speed, cores, ram = host_types["cpu"]
        host = ET.SubElement(zone, "host", id=host_id, speed=speed, core=cores)
        ET.SubElement(host, "prop", id="ram", value=ram) # <-- 修复了内存
        disk = ET.SubElement(host, "disk", id="local_disk", read_bw="200MBps", write_bw="200MBps")
        ET.SubElement(disk, "prop", id="size", value="200GB")
        ET.SubElement(disk, "prop", id="mount", value="/scratch")
        compute_hosts.append(host_id)

    for i in range(num_gpu):
        host_id = f"gpu_host_{i}"
        speed, cores, ram = host_types["gpu"]
        host = ET.SubElement(zone, "host", id=host_id, speed=speed, core=cores)
        ET.SubElement(host, "prop", id="ram", value=ram) # <-- 修复了内存
        disk = ET.SubElement(host, "disk", id="local_disk", read_bw="300MBps", write_bw="300MBps")
        ET.SubElement(disk, "prop", id="size", value="500GB")
        ET.SubElement(disk, "prop", id="mount", value="/scratch")
        compute_hosts.append(host_id)
        
    print(f"  > 添加了 {num_cpu} 个 CPU 节点, {num_gpu} 个 GPU 节点。")

    # 4. 添加网络链接
    link = ET.SubElement(zone, "link", id="network_link", bandwidth="10GBps", latency="1ms")

    # 5. 添加路由 (全连接网络)
    all_host_ids = ["ControllerHost", "StorageHost"] + compute_hosts
    
    # 路由到存储和控制器
    for host in compute_hosts:
        ET.SubElement(zone, "route", src="ControllerHost", dst=host).append(ET.Element("link_ctn", id="network_link"))
        ET.SubElement(zone, "route", src="StorageHost", dst=host).append(ET.Element("link_ctn", id="network_link"))

    # 计算节点之间的路由
    for i in range(len(compute_hosts)):
        for j in range(i + 1, len(compute_hosts)):
            h1 = compute_hosts[i]
            h2 = compute_hosts[j]
            ET.SubElement(zone, "route", src=h1, dst=h2).append(ET.Element("link_ctn", id="network_link"))

    print(f"  > 添加了 {len(all_host_ids) * (len(all_host_ids) - 1) // 2} 条路由。")

    # 6. 写入文件
    tree_str = ET.tostring(platform, encoding='unicode')
    
    # 使用minidom仅格式化platform元素本身
    platform_xml_str = minidom.parseString(tree_str).documentElement.toprettyxml(indent="  ")

    # --- 关键修复：手动添加 XML prolog 和 DOCTYPE ---
    final_xml_content = f"""<?xml version='1.0'?>
<!DOCTYPE platform SYSTEM "https://simgrid.org/simgrid.dtd">
{platform_xml_str}
"""
    # --- 修复结束 ---
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_xml_content)
    print(f"✅ 平台文件已保存: {output_file}")

if __name__ == "__main__":
    # 确保 configs 目录存在
    output_dir = Path("configs")
    output_dir.mkdir(exist_ok=True)
    
    # 生成所有3种规模
    create_platform("small", output_dir / "platform_small.xml")
    create_platform("medium", output_dir / "platform_medium.xml")
    create_platform("large", output_dir / "platform_large.xml")
    
    print("\n🎉 所有平台文件均已生成! 🎉")