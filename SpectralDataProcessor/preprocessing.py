import numpy as np
from scipy.signal import savgol_filter, detrend
from scipy.sparse import csc_matrix, eye, diags
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Preprocessor:
    def __init__(self):
        pass

    # ------------------------------
    # 多样本预处理方法
    # ------------------------------
    def normalize_multi_sample(self, data, method="Min-Max标准化"):
        """对多个样本进行标准化"""
        normalized_samples = {}
        
        for sample_name, intensity in data['samples'].items():
            if method == "Min-Max标准化":
                # 最小-最大标准化
                min_val = np.min(intensity)
                max_val = np.max(intensity)
                if max_val - min_val == 0:
                    normalized = intensity  # 避免除以零
                else:
                    normalized = (intensity - min_val) / (max_val - min_val)
            elif method == "Z-score标准化":
                # Z-score标准化（均值为0，标准差为1）
                mean_val = np.mean(intensity)
                std_val = np.std(intensity)
                if std_val == 0:
                    normalized = intensity - mean_val
                else:
                    normalized = (intensity - mean_val) / std_val
            else:
                normalized = intensity  # 不处理
                
            normalized_samples[sample_name] = normalized
            
        return {
            "wavelength": data['wavelength'],
            "samples": normalized_samples
        }

    def baseline_correction_multi_sample(self, data, lam=100000, p=0.01, n_iter=10):
        """对多个样本进行基线校正（ALS算法）"""
        corrected_samples = {}
        
        for sample_name, intensity in data['samples'].items():
            corrected = self._als_baseline_correction(intensity, lam, p, n_iter)
            corrected_samples[sample_name] = corrected
            
        return {
            "wavelength": data['wavelength'],
            "samples": corrected_samples
        }

    def sg_smoothing_multi_sample(self, data, window_size=11, poly_order=2):
        """对多个样本进行SG平滑"""
        # 确保窗口大小为奇数且不超过数据长度
        if window_size % 2 == 0:
            window_size += 1
        if window_size > len(next(iter(data['samples'].values()))):
            window_size = len(next(iter(data['samples'].values()))) - 1
            if window_size % 2 == 0 and window_size > 1:
                window_size -= 1
        
        smoothed_samples = {}
        for sample_name, intensity in data['samples'].items():
            smoothed = savgol_filter(
                intensity,
                window_length=window_size,
                polyorder=poly_order,
                mode='nearest'
            )
            smoothed_samples[sample_name] = smoothed
            
        return {
            "wavelength": data['wavelength'],
            "samples": smoothed_samples
        }

    # ------------------------------
    # 分析方法
    # ------------------------------
    def quantitative_analysis(self, samples, sample_names, wavelength, concentrations, peak_wavelength):
        """单变量定量分析"""
        # 找到最接近特征峰波长的索引
        peak_idx = np.argmin(np.abs(wavelength - peak_wavelength))
        
        # 提取特征峰处的强度
        intensities = [sample[peak_idx] for sample in samples]
        
        # 计算线性回归（y = ax + b）
        x = np.array(concentrations)
        y = np.array(intensities)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        
        # 计算R²
        y_pred = slope * x + intercept
        ss_total = np.sum((y - np.mean(y)) **2)
        ss_residual = np.sum((y - y_pred)** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        return {
            "方法": "定量分析（单变量）",
            "特征峰波长": peak_wavelength,
            "特征峰索引": peak_idx,
            "浓度值": concentrations,
            "强度值": intensities,
            "斜率": slope,
            "截距": intercept,
            "R²": r_squared,
            "结果": f"校准曲线: y = {slope:.4f}x + {intercept:.4f}, R² = {r_squared:.4f}"
        }

    def qualitative_analysis(self, sample, wavelength, height_threshold=0.2, min_distance=10, sample_name="样本"):
        """定性分析（特征峰识别）"""
        # 归一化强度
        normalized = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        
        # 寻找峰值
        peaks = []
        for i in range(1, len(normalized) - 1):
            # 峰值条件：当前点大于左右邻点，且高度超过阈值
            if (normalized[i] > normalized[i-1] and 
                normalized[i] > normalized[i+1] and 
                normalized[i] > height_threshold):
                
                # 检查与前一个峰的距离
                if not peaks or (i - peaks[-1]['index'] >= min_distance):
                    peaks.append({
                        "index": i,
                        "wavelength": wavelength[i],
                        "intensity": sample[i],
                        "relative_intensity": normalized[i]
                    })
        
        # 格式化特征峰信息
        peak_info = [f"{p['wavelength']:.1f} nm (强度: {p['intensity']:.4f})" for p in peaks]
        
        return {
            "方法": "定性分析（特征峰识别）",
            "样本名": sample_name,
            "波长": wavelength,
            "强度": sample,
            "峰高阈值": height_threshold,
            "最小峰间距": min_distance,
            "特征峰数量": len(peaks),
            "特征峰": peak_info
        }

    def pca_analysis(self, samples, sample_names, n_components=2):
        """主成分分析"""
        # 将样本数据转换为矩阵（样本数×特征数）
        X = np.array(samples)
        
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(X_scaled)
        
        # 整理结果
        pca_data = {}
        for i, name in enumerate(sample_names):
            pc_dict = {f"PC{j+1}": principal_components[i, j] for j in range(n_components)}
            pca_data[name] = pc_dict
        
        # 解释方差率
        explained_variance = [f"PC{i+1}: {var*100:.2f}%" 
                             for i, var in enumerate(pca.explained_variance_ratio_)]
        
        return {
            "方法": "主成分分析(PCA)",
            "主成分数量": n_components,
            "主成分数据": pca_data,
            "解释方差率": explained_variance,
            "累计解释方差率": f"{np.sum(pca.explained_variance_ratio_)*100:.2f}%"
        }

    # ------------------------------
    # 内部辅助方法
    # ------------------------------
    def _als_baseline_correction(self, y, lam=100000, p=0.01, n_iter=10):
        """
        自适应加权迭代最小二乘基线校正
        参考：https://doi.org/10.1016/j.chemolab.2005.10.006
        """
        L = len(y)
        D = csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for _ in range(n_iter):
            W = diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w*y)
            w = p * (y > z) + (1-p) * (y < z)
        return y - z
