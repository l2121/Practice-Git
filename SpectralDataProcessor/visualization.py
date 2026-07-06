import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

class Visualizer:
    def __init__(self):
        self.fig = None  # 存储当前图表
    def plot_spectra(self, plot_data, x_label, y_label, title="多样本光谱图"):
        """
        绘制多条光谱曲线（每个样本一条线）
        :param plot_data: 绘图数据列表，每个元素为{"x":..., "y":..., "name":...}
        """
        self.fig, ax = plt.subplots(figsize=(12, 7))
        
        # 生成不同的颜色和线型（支持更多样本）
        colors = plt.cm.rainbow(np.linspace(0, 1, len(plot_data)))
        linestyles = ['-', '--', '-.', ':'] * (len(plot_data) // 4 + 1)
        
        for i, data in enumerate(plot_data):
            ax.plot(
                data["x"], 
                data["y"], 
                label=data["name"],
                color=colors[i],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1,
                alpha=0.8  # 增加透明度，避免线条重叠时看不清
            )
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # 处理图例（样本过多时自动调整位置）
        if len(plot_data) <= 10:
            ax.legend(loc='best', fontsize=9)
        else:
            # 样本过多时，将图例放在右侧
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
            plt.subplots_adjust(right=0.75)  # 预留图例空间
        
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    
    def plot_calibration_curve(self, x_data, y_data, r_squared, slope, intercept, title="校准曲线"):
        """绘制定量分析校准曲线"""
        self.fig, ax = plt.subplots(figsize=(8, 6))
        
        # 绘制散点
        ax.scatter(x_data, y_data, color='blue', label='样本点', s=50)
        
        # 绘制拟合线
        x_range = np.linspace(min(x_data), max(x_data), 100)
        y_fit = slope * x_range + intercept
        ax.plot(x_range, y_fit, 'r--', label=f'拟合线: y = {slope:.4f}x + {intercept:.4f}')
        
        # 添加R²值
        ax.text(0.05, 0.95, f'R² = {r_squared:.4f}', 
                transform=ax.transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('浓度', fontsize=12)
        ax.set_ylabel('光谱强度', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_with_peaks(self, x_data, y_data, peaks, x_label="波长 (nm)", y_label="强度", title="带特征峰的光谱图"):
        """绘制带有特征峰标记的光谱图"""
        self.fig, ax = plt.subplots(figsize=(10, 6))
        
        # 绘制光谱
        ax.plot(x_data, y_data, color='blue', linewidth=1.5)
        
        # 标记特征峰
        peak_values = []
        peak_positions = []
        for peak_str in peaks:
            # 从字符串中提取波长数值
            peak_val = float(peak_str.split()[0])
            # 找到最接近的x数据点
            idx = np.argmin(np.abs(x_data - peak_val))
            peak_values.append(y_data[idx])
            peak_positions.append(peak_val)
        
        ax.scatter(peak_positions, peak_values, color='red', s=60, zorder=5, label='特征峰')
        
        # 在峰上方标注波长
        for x, y in zip(peak_positions, peak_values):
            ax.annotate(f'{x:.1f}', (x, y), xytext=(0, 10), 
                        textcoords='offset points', ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    
    def plot_pca_results(self, pca_data, explained_variance, title="PCA分析结果"):
        """绘制PCA结果散点图"""
        # 提取PC1和PC2数据
        samples = list(pca_data.keys())
        pc1 = [pca_data[s]["PC1"] for s in samples]
        pc2 = [pca_data[s]["PC2"] for s in samples]
        
        self.fig, ax = plt.subplots(figsize=(9, 7))
        
        # 绘制散点图
        scatter = ax.scatter(pc1, pc2, c=range(len(samples)), cmap='viridis', s=100, alpha=0.8)
        
        # 添加样本标签
        for i, sample in enumerate(samples):
            ax.annotate(sample, (pc1[i], pc2[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # 添加解释方差率到轴标签
        pc1_var = explained_variance[0].split(": ")[1] if len(explained_variance) > 0 else ""
        pc2_var = explained_variance[1].split(": ")[1] if len(explained_variance) > 1 else ""
        
        ax.set_xlabel(f'主成分1 ({pc1_var})', fontsize=12)
        ax.set_ylabel(f'主成分2 ({pc2_var})', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label('样本索引')
        
        plt.tight_layout()
        plt.show()
    
    def save_plot(self, save_path):
        """保存当前图表"""
        try:
            if self.fig is not None:
                self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
                return True
            return False
        except Exception as e:
            print(f"保存图表失败: {e}")
            return False
