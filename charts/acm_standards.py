#!/usr/bin/env python3
"""
ACM论文图表标准配置和验证工具
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Any
import os

class ACMChartStandards:
    """ACM论文图表标准配置"""
    
    # ACM推荐的图表尺寸 (英寸)
    FIGURE_SIZES = {
        'single_column': (3.5, 2.625),    # 单栏图 (宽高比4:3)
        'double_column': (7.16, 5.37),    # 双栏图
        'square': (3.5, 3.5),             # 正方形图
        'wide': (7.16, 3.0),              # 宽图
        'tall': (3.5, 5.0)                # 高图
    }
    
    # ACM配色方案 (色盲友好 + 黑白打印友好)
    COLORS = {
        'primary': '#0173B2',      # 深蓝
        'secondary': '#DE8F05',    # 橙色
        'tertiary': '#029E73',     # 绿色
        'quaternary': '#CC78BC',   # 粉色
        'quinary': '#CA9161',      # 棕色
        'text': '#000000',         # 纯黑文字
        'grid': '#CCCCCC',         # 浅灰网格
        'background': '#FFFFFF'    # 纯白背景
    }
    
    # 算法专用颜色
    ALGORITHM_COLORS = {
        'WASS-RAG': '#0173B2',     # 深蓝 - 主要方法
        'WASS-DRL': '#DE8F05',     # 橙色 - DRL基线
        'HEFT': '#029E73',         # 绿色 - 传统启发式
        'FIFO': '#CC78BC',         # 粉色 - 简单方法
        'SJF': '#CA9161',          # 棕色 - 另一基线
        'Random': '#999999'        # 灰色 - 随机基线
    }
    
    @staticmethod
    def configure_matplotlib_for_acm():
        """配置matplotlib以符合ACM标准"""
        
        # 基础配置
        plt.rcParams.update({
            # 字体配置 - ACM标准
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.titlesize': 11,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 12,
            
            # 图形质量 - 出版级别
            'figure.dpi': 300,
            'savefig.dpi': 600,
            'savefig.format': 'pdf',
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'savefig.transparent': False,
            
            # 线条和标记
            'lines.linewidth': 1.2,
            'lines.markersize': 4,
            'patch.linewidth': 0.5,
            'axes.linewidth': 0.6,
            
            # 网格设置
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'axes.axisbelow': True,
            
            # 布局优化
            'figure.constrained_layout.use': True,
            'axes.unicode_minus': False,
            'text.usetex': False,  # 除非确实需要LaTeX
            
            # 颜色和样式
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.edgecolor': 'black',
            'xtick.color': 'black',
            'ytick.color': 'black',
            'text.color': 'black'
        })
        
        print("✅ Matplotlib configured for ACM publication standards")
    
    @staticmethod
    def validate_figure_for_acm(fig, chart_type: str = "unknown") -> Dict[str, Any]:
        """验证图表是否符合ACM标准"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # 检查图形尺寸
        fig_width, fig_height = fig.get_size_inches()
        
        if fig_width > 7.2:
            validation_results['errors'].append(f"Width {fig_width:.2f}\" exceeds ACM limit (7.16\")")
            validation_results['valid'] = False
        
        if fig_height > 9.5:
            validation_results['errors'].append(f"Height {fig_height:.2f}\" exceeds ACM limit (9.5\")")
            validation_results['valid'] = False
        
        # 检查字体大小
        for ax in fig.get_axes():
            # 检查标题字体
            title = ax.get_title()
            if title and hasattr(ax.title, 'get_fontsize'):
                title_size = ax.title.get_fontsize()
                if title_size < 10 or title_size > 12:
                    validation_results['warnings'].append(f"Title font size {title_size} not in recommended range (10-12)")
            
            # 检查轴标签字体
            xlabel_size = ax.xaxis.label.get_fontsize()
            ylabel_size = ax.yaxis.label.get_fontsize()
            
            if xlabel_size < 9 or xlabel_size > 11:
                validation_results['warnings'].append(f"X-axis label font size {xlabel_size} not in recommended range (9-11)")
            
            if ylabel_size < 9 or ylabel_size > 11:
                validation_results['warnings'].append(f"Y-axis label font size {ylabel_size} not in recommended range (9-11)")
        
        # 推荐建议
        if chart_type == "heatmap":
            validation_results['recommendations'].extend([
                "Consider using colorbrewer palettes for better accessibility",
                "Ensure colorbar labels are clearly readable",
                "Test appearance in grayscale for print compatibility"
            ])
        elif chart_type == "line":
            validation_results['recommendations'].extend([
                "Use different line styles (solid, dashed, dotted) for B&W compatibility",
                "Ensure markers are distinguishable in grayscale",
                "Limit to 5-6 lines maximum for clarity"
            ])
        elif chart_type == "bar":
            validation_results['recommendations'].extend([
                "Use patterns/hatching for B&W accessibility",
                "Ensure adequate spacing between bars",
                "Consider horizontal bars for long labels"
            ])
        
        return validation_results
    
    @staticmethod
    def save_acm_figure(fig, filepath: str, chart_type: str = "unknown"):
        """保存符合ACM标准的图表文件"""
        
        # 验证图表
        validation = ACMChartStandards.validate_figure_for_acm(fig, chart_type)
        
        if not validation['valid']:
            print("⚠️  Figure validation failed:")
            for error in validation['errors']:
                print(f"   ❌ {error}")
        
        if validation['warnings']:
            print("⚠️  Figure validation warnings:")
            for warning in validation['warnings']:
                print(f"   ⚠️  {warning}")
        
        # 保存多种格式
        base_path = os.path.splitext(filepath)[0]
        
        # PDF - ACM首选格式
        pdf_path = f"{base_path}.pdf"
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight', 
                   pad_inches=0.02, transparent=False)
        
        # PNG - 高分辨率备用
        png_path = f"{base_path}.png"
        fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight',
                   pad_inches=0.02, transparent=False)
        
        # EPS - 某些会议要求
        eps_path = f"{base_path}.eps"
        fig.savefig(eps_path, format='eps', bbox_inches='tight',
                   pad_inches=0.02, transparent=False)
        
        print(f"✅ ACM-compliant figures saved:")
        print(f"   📄 PDF: {pdf_path}")
        print(f"   🖼️  PNG: {png_path}")
        print(f"   📐 EPS: {eps_path}")
        
        if validation['recommendations']:
            print("\n💡 Recommendations for improvement:")
            for rec in validation['recommendations']:
                print(f"   • {rec}")
        
        return pdf_path

def create_acm_colormap():
    """创建ACM友好的colormap"""
    from matplotlib.colors import LinearSegmentedColormap
    
    # 蓝色系colormap (单色渐变，适合热力图)
    blues_acm = LinearSegmentedColormap.from_list(
        'blues_acm',
        ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#084594'],
        N=256
    )
    
    # 注册colormap
    plt.register_cmap(cmap=blues_acm)
    
    return blues_acm

def main():
    """演示ACM标准配置"""
    print("🎯 ACM Paper Chart Standards Configuration")
    print("=" * 50)
    
    # 配置matplotlib
    ACMChartStandards.configure_matplotlib_for_acm()
    
    # 创建ACM colormap
    create_acm_colormap()
    
    # 显示配置信息
    print(f"📐 Recommended figure sizes:")
    for name, size in ACMChartStandards.FIGURE_SIZES.items():
        print(f"   • {name}: {size[0]}\" × {size[1]}\"")
    
    print(f"\n🎨 ACM color palette:")
    for name, color in ACMChartStandards.ALGORITHM_COLORS.items():
        print(f"   • {name}: {color}")
    
    print(f"\n✅ Ready for ACM-quality chart generation!")

if __name__ == "__main__":
    main()
