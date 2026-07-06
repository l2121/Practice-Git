import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class Analyzer:
    def detect_peaks(self, wavelength, intensity, height=None, distance=10):
        """
        特征峰识别
        :param wavelength: 波长数据
        :param intensity: 强度数据
        :param height: 峰高阈值
        :param distance: 峰间距（单位：数据点）
        :return: 特征峰列表（[(峰位nm, 峰高AU), ...]）
        """
        # 识别峰
        peaks_idx, _ = find_peaks(intensity, height=height, distance=distance)
        
        # 按峰高排序，取前10个最强峰
        if len(peaks_idx) > 10:
            peak_heights = intensity[peaks_idx]
            sorted_indices = np.argsort(peak_heights)[::-1][:10]  # 降序排列，取前10
            peaks_idx = peaks_idx[sorted_indices]
        
        peaks = [(wavelength[idx], intensity[idx]) for idx in peaks_idx]
        return peaks

    def single_variable_calibration(self, concentrations, intensities):
        """
        单变量定量分析（朗伯-比尔定律）
        :param concentrations: 标准样品浓度（列表/数组）
        :param intensities: 标准样品对应强度（列表/数组）
        :return: 校准结果（r2、斜率、截距）
        """
        # 转换为数组并reshape（适配sklearn）
        X = np.array(concentrations).reshape(-1, 1)
        y = np.array(intensities).reshape(-1, 1)
        
        if len(X) < 2:
            raise ValueError("至少需要2个标准样品进行校准")
        
        # 线性回归
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算R²
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        return {
            "r2": r2,
            "slope": model.coef_[0][0],
            "intercept": model.intercept_[0],
            "y_pred": y_pred.flatten()
        }
