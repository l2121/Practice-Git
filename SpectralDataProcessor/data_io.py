import pandas as pd
import os
import numpy as np

class DataIO:
    def __init__(self):
        self.supported_formats = {
            '.csv': self._read_text,
            '.txt': self._read_text,
            '.xlsx': self._read_excel,
            '.xls': self._read_excel
        }
        self.common_encodings = ['utf-8-sig', 'gbk', 'gb2312', 'latin-1']

    def load_data(self, file_path):
        """兼容旧版本的单样本加载方法（保留用于向后兼容）"""
        return self.load_multi_sample_data(file_path)

    def load_multi_sample_data(self, file_path):
        """
        加载包含多个样本的光谱数据，支持波长在列或行两种格式
        返回格式: {
            "wavelength": 波长数组,
            "samples": {样本名1: 强度数组, 样本名2: 强度数组, ...}
        }
        """
        try:
            ext = self._get_file_extension(file_path).lower()
            if ext not in self.supported_formats:
                print(f"不支持的文件类型: {ext}")
                return None

            # 读取原始数据
            df = self.supported_formats[ext](file_path)
            if df is None:
                return None

            # 处理表头（跳过非数值行）
            data_df = self._remove_header_rows(df)
            if data_df is None:
                return None

            # 清洗数据
            numeric_df = self._clean_data(data_df)
            if numeric_df is None:
                return None

            # 提取多样本数据（支持列和行两种格式）
            result = self._extract_multi_sample_data(numeric_df, file_path)
            return result

        except Exception as e:
            print(f"多样本数据加载失败：{str(e)}")
            return None

    def _extract_multi_sample_data(self, df, file_path):
        """提取多个样本的数据（支持波长在列或行中）"""
        try:
            # 尝试判断波长数据位置（列或行）
            # 检查第一列是否为波长（数值连续且递增）
            first_col = df.iloc[:, 0].values
            if self._is_wavelength_data(first_col):
                # 波长在第一列，其余列为样本
                wavelength = first_col.astype(float)
                samples = {}
                for col_idx in range(1, df.shape[1]):
                    sample_name = self._get_sample_name(df, col_idx, file_path, is_row=False)
                    intensity = df.iloc[:, col_idx].values.astype(float)
                    samples[sample_name] = intensity
            else:
                # 检查第一行是否为波长
                first_row = df.iloc[0, :].values
                if self._is_wavelength_data(first_row):
                    wavelength = first_row.astype(float)
                    samples = {}
                    for row_idx in range(1, df.shape[0]):
                        sample_name = self._get_sample_name(df, row_idx, file_path, is_row=True)
                        intensity = df.iloc[row_idx, :].values.astype(float)
                        samples[sample_name] = intensity
                else:
                    raise ValueError("无法识别波长数据，请检查数据格式")

            if not samples:
                print("未找到有效的样本数据")
                return None

            return {
                "wavelength": wavelength,
                "samples": samples
            }
        except Exception as e:
            print(f"提取多样本数据失败：{str(e)}")
            return None

    def _is_wavelength_data(self, data):
        """判断数据是否为波长（连续递增且差值合理）"""
        # 过滤非数值
        numeric_data = []
        for val in data:
            if self._is_numeric(val):
                numeric_data.append(float(val))
            else:
                return False
                
        if len(numeric_data) < 3:
            return False
            
        # 转换为numpy数组
        numeric_data = np.array(numeric_data)
        
        # 检查是否递增
        if not np.all(np.diff(numeric_data) > 0):
            return False
            
        # 检查波长范围是否合理（200-2500nm之间）
        if not (np.min(numeric_data) >= 200 and np.max(numeric_data) <= 2500):
            return False
            
        # 检查波长间隔是否在合理范围（0.1-10nm）
        diffs = np.diff(numeric_data)
        if not (np.min(diffs) >= 0.1 and np.max(diffs) <= 10):
            return False
            
        return True

    def _get_sample_name(self, df, idx, file_path, is_row=False):
        """获取样本名称（支持行和列两种格式）"""
        try:
            base_name = self._get_base_filename(file_path)
            if is_row:
                # 行格式时尝试从索引获取样本名
                if isinstance(df.index[idx], str) and df.index[idx].strip():
                    return f"{base_name}_{df.index[idx].strip()}"
                else:
                    return f"{base_name}_样本{idx}"
            else:
                # 列格式时尝试从列名获取样本名
                col_name = df.columns[idx]
                if isinstance(col_name, str) and col_name.strip():
                    return f"{base_name}_{col_name.strip()}"
                else:
                    return f"{base_name}_样本{idx}"
        except:
            return f"{self._get_base_filename(file_path)}_样本{idx}"

    def _get_base_filename(self, file_path):
        """从文件路径中提取基础文件名（不含扩展名）"""
        return os.path.splitext(os.path.basename(file_path))[0]

    def _remove_header_rows(self, df):
        """自动识别并跳过表头行"""
        start_row = 0
        for i in range(len(df)):
            row = df.iloc[i]
            if all(self._is_numeric(x) for x in row):
                start_row = i
                break
        data_df = df.iloc[start_row:].reset_index(drop=True)
        return data_df if not data_df.empty else None

    def _is_numeric(self, value):
        """判断值是否为数值"""
        if pd.isna(value):
            return False
        try:
            float(str(value).strip())
            return True
        except:
            return False

    def _get_file_extension(self, file_path):
        return os.path.splitext(file_path)[1]

    def _read_text(self, file_path):
        """读取文本文件（CSV/TXT）"""
        for encoding in self.common_encodings:
            for sep in [',', '\t', ';', '|']:
                try:
                    df = pd.read_csv(
                        file_path,
                        sep=sep,
                        header=None,
                        engine='python',
                        encoding=encoding,
                        skip_blank_lines=True
                    )
                    if not df.empty:
                        return df
                except:
                    continue
        return None

    def _read_excel(self, file_path):
        """读取Excel文件"""
        try:
            return pd.read_excel(file_path, header=None)
        except:
            return None

    def _clean_data(self, df):
        """清洗数据"""
        try:
            numeric_df = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
            numeric_df = numeric_df.dropna(axis=0, how='all').dropna(axis=1, how='all')
            return numeric_df if numeric_df.shape[1] >= 2 else None
        except:
            return None

    def save_data(self, data_dict, save_path):
        """保存多样本数据"""
        try:
            if not data_dict or "wavelength" not in data_dict or "samples" not in data_dict:
                return False
            export_data = {"波长": data_dict["wavelength"]}
            export_data.update(data_dict["samples"])
            pd.DataFrame(export_data).to_csv(save_path, index=False, encoding="utf-8-sig")
            return True
        except Exception as e:
            print(f"保存数据失败：{e}")
            return False