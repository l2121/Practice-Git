import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
from preprocessing import Preprocessor
from visualization import Visualizer
from data_io import DataIO

class SpectralAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("多光谱数据特征提取与建模分析软件")
        self.root.geometry("1200x800")
        
        # --------------------------
        # 添加全局字体设置（关键代码）
        # --------------------------
        default_font = ('SimHei', 14)  # 字体名称+大小，12可根据需要增大
        # 配置所有Tkinter控件的默认字体
        self.root.option_add("*Font", default_font)
        # 单独调整列表框和文本框的字体（可选，如需更大）
        self.root.option_add("*Listbox.Font", ('SimHei', 13))
        self.root.option_add("*Text.Font", ('SimHei', 13))

        # 数据存储 - 适配多样本结构
        self.data_dict = {}  # {文件名: {"wavelength":..., "samples": {样本名: 强度数组,...}}}
        self.processed_data = None
        self.analysis_results = None
        
        # 初始化工具类
        self.data_io = DataIO()
        self.preprocessor = Preprocessor()
        self.visualizer = Visualizer()
        
        # 创建界面
        self._create_widgets()
        
    def _create_widgets(self):
        # 创建标签页控件
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
         # 设置标签页标题字体
        style = ttk.Style()
        style.configure("TNotebook.Tab", 
                        font=('SimHei', 14),  # 字体大小14，可改为16、18等
                        padding=[15, 5])  # 标题的内边距
       
        # 1. 数据导入标签页
        import_frame = ttk.Frame(notebook)
        notebook.add(import_frame, text="数据导入")
        self._setup_import_tab(import_frame)
        
        # 2. 数据预处理标签页
        preprocess_frame = ttk.Frame(notebook)
        notebook.add(preprocess_frame, text="数据预处理")
        self._setup_preprocess_tab(preprocess_frame)
        
        # 3. 光谱分析标签页
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="光谱分析")
        self._setup_analysis_tab(analysis_frame)
        
        # 4. 结果可视化标签页
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="结果可视化")
        self._setup_visualization_tab(viz_frame)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # ------------------------------
    # 1. 数据导入标签页
    # ------------------------------
    def _setup_import_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="数据文件", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 导入按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(btn_frame, text="导入光谱数据", command=self._import_data).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="清除数据", command=self._clear_data).pack(side=tk.LEFT, padx=5)
        
        # 文件列表
        ttk.Label(frame, text="已导入文件:").pack(anchor=tk.W)
        self.file_listbox = tk.Listbox(frame, height=5)
        self.file_listbox.pack(fill=tk.X, pady=5)
        
        # 样本列表（新增：显示当前文件包含的样本）
        ttk.Label(frame, text="包含样本:").pack(anchor=tk.W)
        self.sample_listbox = tk.Listbox(frame, height=8)
        self.sample_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 数据信息
        info_frame = ttk.Frame(frame)
        info_frame.pack(fill=tk.X, pady=10)
        self.data_info = scrolledtext.ScrolledText(info_frame, height=5, wrap=tk.WORD)
        self.data_info.pack(fill=tk.BOTH, expand=True)
        self.data_info.insert(tk.END, "数据信息将显示在这里\n提示：每个文件可能包含多个样本")
        self.data_info.config(state=tk.DISABLED)
        
        # 绑定文件选择事件，显示对应样本
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)
    
    # ------------------------------
    # 2. 数据预处理标签页
    # ------------------------------
    def _setup_preprocess_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="预处理设置", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 预处理方法选择
        method_frame = ttk.Frame(frame)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(method_frame, text="选择预处理方法:").pack(side=tk.LEFT)
        self.method_combobox = ttk.Combobox(
            method_frame, 
            values=["标准化", "基线校正(ALS)", "SG平滑"],
            width=18
        )
        self.method_combobox.current(0)
        self.method_combobox.pack(side=tk.LEFT, padx=5)
        
        # 参数区域
        self.param_frame = ttk.LabelFrame(frame, text="参数设置")
        self.param_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # 创建参数控件
        self.pre_param1_label = ttk.Label(self.param_frame, text="")
        self.pre_param1_entry = ttk.Entry(self.param_frame, width=15)
        self.pre_param2_label = ttk.Label(self.param_frame, text="")
        self.pre_param2_entry = ttk.Entry(self.param_frame, width=15)
        
        # 执行按钮
        ttk.Button(frame, text="执行预处理", command=self._run_preprocessing).pack(pady=10)
        
        # 预处理结果
        result_frame = ttk.LabelFrame(frame, text="预处理状态")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.preprocess_result = scrolledtext.ScrolledText(result_frame, height=5, wrap=tk.WORD)
        self.preprocess_result.pack(fill=tk.BOTH, expand=True)
        self.preprocess_result.insert(tk.END, "预处理结果将显示在这里")
        self.preprocess_result.config(state=tk.DISABLED)
        
        # 绑定方法切换事件
        self.method_combobox.bind("<<ComboboxSelected>>", self._update_preprocess_params)
        self._update_preprocess_params(None)
    
    # ------------------------------
    # 3. 光谱分析标签页
    # ------------------------------
    def _setup_analysis_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="分析设置", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 分析方法选择
        method_frame = ttk.Frame(frame)
        method_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(method_frame, text="选择分析方法:").pack(side=tk.LEFT)
        self.analysis_combobox = ttk.Combobox(
            method_frame, 
            values=["定量分析（单变量）", "定性分析（特征峰识别）", "主成分分析(PCA)"],
            width=22
        )
        self.analysis_combobox.current(0)
        self.analysis_combobox.pack(side=tk.LEFT, padx=5)
        
        # 参数区域
        self.analysis_param_frame = ttk.LabelFrame(frame, text="参数设置")
        self.analysis_param_frame.pack(fill=tk.X, pady=(0, 10), padx=5)
        
        # 创建分析参数控件
        self.ana_param1_label = ttk.Label(self.analysis_param_frame, text="")
        self.ana_param1_entry = ttk.Entry(self.analysis_param_frame, width=20)
        self.ana_param2_label = ttk.Label(self.analysis_param_frame, text="")
        self.ana_param2_entry = ttk.Entry(self.analysis_param_frame, width=20)
        self.ana_param3_label = ttk.Label(self.analysis_param_frame, text="")
        self.ana_param3_entry = ttk.Entry(self.analysis_param_frame, width=20)
        
        # 执行按钮
        ttk.Button(frame, text="执行分析", command=self._run_analysis).pack(pady=10)
        
        # 分析结果
        result_frame = ttk.LabelFrame(frame, text="分析结果")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.analysis_result = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD)
        self.analysis_result.pack(fill=tk.BOTH, expand=True)
        self.analysis_result.insert(tk.END, "分析结果将显示在这里")
        self.analysis_result.config(state=tk.DISABLED)
        
        # 绑定方法切换事件
        self.analysis_combobox.bind("<<ComboboxSelected>>", self._update_analysis_params)
        self._update_analysis_params(None)
    
    # ------------------------------
    # 4. 可视化标签页
    # ------------------------------
    def _setup_visualization_tab(self, parent):
        frame = ttk.LabelFrame(parent, text="光谱图绘制", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 绘图按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="绘制原始光谱", command=self._plot_original).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="绘制处理后光谱", command=self._plot_processed).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="绘制分析结果", command=self._plot_analysis).pack(side=tk.LEFT, padx=5)
        
        # 样本筛选（新增：可选择显示哪些样本）
        filter_frame = ttk.LabelFrame(frame, text="样本筛选")
        filter_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(filter_frame, text="按品种筛选（空表示全部）:").pack(side=tk.LEFT)
        self.variety_filter = ttk.Entry(filter_frame, width=20)
        self.variety_filter.pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame, text="应用筛选", command=self._update_plot_filters).pack(side=tk.LEFT, padx=5)
        
        # 绘图选项
        option_frame = ttk.LabelFrame(frame, text="绘图选项")
        option_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(option_frame, text="图表标题:").pack(side=tk.LEFT)
        self.plot_title = ttk.Entry(option_frame, width=30)
        self.plot_title.insert(0, "多样本光谱对比图")
        self.plot_title.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(option_frame, text="X轴标签:").pack(side=tk.LEFT, padx=5)
        self.x_label = ttk.Entry(option_frame, width=15)
        self.x_label.insert(0, "波长 (nm)")
        self.x_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(option_frame, text="Y轴标签:").pack(side=tk.LEFT, padx=5)
        self.y_label = ttk.Entry(option_frame, width=15)
        self.y_label.insert(0, "强度")
        self.y_label.pack(side=tk.LEFT, padx=5)
        
        # 保存图表按钮
        ttk.Button(btn_frame, text="保存当前图表", command=self._save_plot).pack(side=tk.RIGHT)
        
        # 当前筛选条件
        self.current_filter = ""  # 修改1：不再存储小写的筛选条件，而是在匹配时统一处理
    
    # ------------------------------
    # 预处理参数动态更新
    # ------------------------------
    def _update_preprocess_params(self, event):
        for widget in self.param_frame.winfo_children():
            widget.pack_forget()
        
        method = self.method_combobox.get()
        
        if method == "标准化":
            self.pre_param1_label.config(text="标准化方式:")
            self.pre_param1_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.pre_param1_entry.delete(0, tk.END)
            self.pre_param1_entry.insert(0, "Min-Max标准化")
            self.pre_param1_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
        elif method == "基线校正(ALS)":
            self.pre_param1_label.config(text="平滑参数(lam):")
            self.pre_param1_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.pre_param1_entry.delete(0, tk.END)
            self.pre_param1_entry.insert(0, "100000")
            self.pre_param1_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
            self.pre_param2_label.config(text="权重参数(p):")
            self.pre_param2_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.pre_param2_entry.delete(0, tk.END)
            self.pre_param2_entry.insert(0, "0.01")
            self.pre_param2_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
        elif method == "SG平滑":
            self.pre_param1_label.config(text="窗口大小(奇数):")
            self.pre_param1_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.pre_param1_entry.delete(0, tk.END)
            self.pre_param1_entry.insert(0, "9")  # 优化后的参数
            self.pre_param1_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
            self.pre_param2_label.config(text="多项式阶数:")
            self.pre_param2_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.pre_param2_entry.delete(0, tk.END)
            self.pre_param2_entry.insert(0, "2")  # 优化后的参数
            self.pre_param2_entry.pack(side=tk.LEFT, padx=5, pady=5)
    
    # ------------------------------
    # 光谱分析参数动态更新
    # ------------------------------
    def _update_analysis_params(self, event):
        for widget in self.analysis_param_frame.winfo_children():
            widget.pack_forget()
        
        method = self.analysis_combobox.get()
        
        if method == "定量分析（单变量）":
            self.ana_param1_label.config(text="标准浓度值(逗号分隔):")
            self.ana_param1_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.ana_param1_entry.delete(0, tk.END)
            self.ana_param1_entry.insert(0, "1.0,2.0,3.0,4.0,5.0")
            self.ana_param1_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
            self.ana_param2_label.config(text="特征峰波长(nm):")
            self.ana_param2_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.ana_param2_entry.delete(0, tk.END)
            self.ana_param2_entry.insert(0, "2100")
            self.ana_param2_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
        elif method == "定性分析（特征峰识别）":
            self.ana_param1_label.config(text="峰高阈值(相对值):")
            self.ana_param1_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.ana_param1_entry.delete(0, tk.END)
            self.ana_param1_entry.insert(0, "0.2")
            self.ana_param1_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
            self.ana_param2_label.config(text="最小峰间距(数据点):")
            self.ana_param2_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.ana_param2_entry.delete(0, tk.END)
            self.ana_param2_entry.insert(0, "10")
            self.ana_param2_entry.pack(side=tk.LEFT, padx=5, pady=5)
            
        elif method == "主成分分析(PCA)":
            self.ana_param1_label.config(text="主成分数量:")
            self.ana_param1_label.pack(side=tk.LEFT, padx=5, pady=5)
            self.ana_param1_entry.delete(0, tk.END)
            self.ana_param1_entry.insert(0, "2")
            self.ana_param1_entry.pack(side=tk.LEFT, padx=5, pady=5)
    
    # ------------------------------
    # 数据导入功能（核心修改：支持多样本）
    # ------------------------------
    def _import_data(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("光谱文件", "*.csv;*.txt;*.xlsx;*.xls")]
        )
        if not file_paths:
            return
            
        for path in file_paths:
            file_name = os.path.basename(path)
            if file_name not in self.data_dict:  # 避免重复导入
                # 调用DataIO加载多样本数据
                data = self.data_io.load_multi_sample_data(path)
                if data:
                    self.data_dict[file_name] = data
                    self.file_listbox.insert(tk.END, file_name)
                    self.status_var.set(f"已导入: {file_name}（包含{len(data['samples'])}个样本）")
        
        # 更新数据信息
        self._update_data_info()
    
    def _clear_data(self):
        self.data_dict = {}
        self.processed_data = None
        self.file_listbox.delete(0, tk.END)
        self.sample_listbox.delete(0, tk.END)
        self._update_data_info()
        self.status_var.set("已清除所有数据")
    
    def _on_file_select(self, event):
        """选择文件后显示包含的样本"""
        selected_indices = self.file_listbox.curselection()
        if not selected_indices:
            return
            
        selected_file = self.file_listbox.get(selected_indices[0])
        if selected_file in self.data_dict:
            self.sample_listbox.delete(0, tk.END)
            # 显示该文件包含的所有样本
            for sample_name in self.data_dict[selected_file]['samples'].keys():
                self.sample_listbox.insert(tk.END, sample_name)
    
    def _update_data_info(self):
        self.data_info.config(state=tk.NORMAL)
        self.data_info.delete(1.0, tk.END)
        
        if not self.data_dict:
            self.data_info.insert(tk.END, "没有导入数据")
        else:
            total_samples = 0
            self.data_info.insert(tk.END, f"共导入 {len(self.data_dict)} 个文件\n\n")
            for name, data in self.data_dict.items():
                sample_count = len(data['samples'])
                total_samples += sample_count
                self.data_info.insert(tk.END, f"{name}:\n")
                self.data_info.insert(tk.END, f"  波长范围: {min(data['wavelength']):.1f} - {max(data['wavelength']):.1f} nm\n")
                self.data_info.insert(tk.END, f"  数据点数量: {len(data['wavelength'])}\n")
                self.data_info.insert(tk.END, f"  样本数量: {sample_count}\n\n")
            self.data_info.insert(tk.END, f"总计样本数量: {total_samples}")
        
        self.data_info.config(state=tk.DISABLED)
    
    def _update_plot_filters(self):
        """更新样本筛选条件 - 修改2：不再将筛选条件转为小写"""
        self.current_filter = self.variety_filter.get().strip()  # 保留原始大小写
        self.status_var.set(f"已应用筛选: {self.current_filter or '显示所有样本'}")
    
    # ------------------------------
    # 预处理功能（支持多样本）
    # ------------------------------
    def _run_preprocessing(self):
        if not self.data_dict:
            messagebox.showwarning("警告", "请先导入数据")
            return
            
        method = self.method_combobox.get()
        try:
            # 对所有文件的所有样本执行预处理
            self.processed_data = {}
            total_samples = 0
            
            for file_name, data in self.data_dict.items():
                if method == "标准化":
                    norm_method = self.pre_param1_entry.get()
                    processed = self.preprocessor.normalize_multi_sample(
                        data, norm_method
                    )
                    result_text = f"标准化完成，方法: {norm_method}"
                    
                elif method == "基线校正(ALS)":
                    lam = float(self.pre_param1_entry.get())
                    p = float(self.pre_param2_entry.get())
                    processed = self.preprocessor.baseline_correction_multi_sample(
                        data, lam, p
                    )
                    result_text = f"基线校正完成，参数: lam={lam}, p={p}"
                    
                elif method == "SG平滑":
                    window = int(self.pre_param1_entry.get())
                    order = int(self.pre_param2_entry.get())
                    processed = self.preprocessor.sg_smoothing_multi_sample(
                        data, window, order
                    )
                    result_text = f"SG平滑完成，参数: 窗口大小={window}, 多项式阶数={order}"
                
                self.processed_data[file_name] = processed
                total_samples += len(processed['samples'])
            
            self.status_var.set(f"预处理完成: {method}（处理{total_samples}个样本）")
            self.preprocess_result.config(state=tk.NORMAL)
            self.preprocess_result.delete(1.0, tk.END)
            self.preprocess_result.insert(tk.END, f"{result_text}\n共处理 {total_samples} 个样本")
            self.preprocess_result.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("错误", f"预处理失败: {str(e)}")
            self.status_var.set("预处理失败")
    
    # ------------------------------
    # 分析功能（支持多样本）
    # ------------------------------
    def _run_analysis(self):
        if not self.processed_data and not self.data_dict:
            messagebox.showwarning("警告", "请先导入并预处理数据")
            return
            
        # 收集所有样本数据（处理后的数据优先）
        data_to_analyze = {}
        if self.processed_data:
            for file_name, data in self.processed_data.items():
                data_to_analyze[file_name] = data
        else:
            for file_name, data in self.data_dict.items():
                data_to_analyze[file_name] = data
        
        method = self.analysis_combobox.get()
        try:
            all_samples = []
            all_sample_names = []
            # 提取所有样本数据
            for file_data in data_to_analyze.values():
                for sample_name, intensity in file_data['samples'].items():
                    all_samples.append(intensity)
                    all_sample_names.append(sample_name)
            
            if method == "定量分析（单变量）":
                conc_str = self.ana_param1_entry.get()
                concentrations = [float(x.strip()) for x in conc_str.split(',')]
                peak_wavelength = float(self.ana_param2_entry.get())
                
                # 确保浓度数量与样本数量一致
                if len(concentrations) != len(all_samples):
                    raise ValueError(f"浓度数量({len(concentrations)})与样本数量({len(all_samples)})不一致")
                
                # 执行定量分析
                self.analysis_results = self.preprocessor.quantitative_analysis(
                    all_samples, all_sample_names, 
                    file_data['wavelength'],  # 使用第一个文件的波长
                    concentrations, peak_wavelength
                )
                
            elif method == "定性分析（特征峰识别）":
                height_threshold = float(self.ana_param1_entry.get())
                min_distance = int(self.ana_param2_entry.get())
                
                # 执行定性分析（以第一个样本为例，实际可扩展为所有样本）
                self.analysis_results = self.preprocessor.qualitative_analysis(
                    all_samples[0], file_data['wavelength'],
                    height_threshold, min_distance,
                    sample_name=all_sample_names[0]
                )
                
            elif method == "主成分分析(PCA)":
                n_components = int(self.ana_param1_entry.get())
                
                # 执行PCA分析
                self.analysis_results = self.preprocessor.pca_analysis(
                    all_samples, all_sample_names, n_components
                )
                
            self.status_var.set(f"分析完成: {method}（分析{len(all_samples)}个样本）")
            self.analysis_result.config(state=tk.NORMAL)
            self.analysis_result.delete(1.0, tk.END)
            for key, value in self.analysis_results.items():
                self.analysis_result.insert(tk.END, f"{key}: {value}\n")
            self.analysis_result.config(state=tk.DISABLED)
            
        except Exception as e:
            messagebox.showerror("错误", f"分析失败: {str(e)}")
            self.status_var.set("分析失败")
    
    # ------------------------------
    # 可视化功能（核心修改：显示多条曲线）
    # ------------------------------
    def _get_plot_data(self, use_processed=False):
        """获取符合筛选条件的绘图数据 - 修改3：实现不区分大小写的匹配逻辑"""
        plot_data = []
        
        # 选择要绘制的数据（原始或处理后）
        data_source = self.processed_data if use_processed else self.data_dict
        if not data_source:
            return None
            
        # 获取筛选关键词并转为小写用于不区分大小写的匹配
        filter_keyword = self.current_filter.lower()
        
        # 遍历所有文件和样本
        for file_data in data_source.values():
            wavelength = file_data['wavelength']
            for sample_name, intensity in file_data['samples'].items():
                # 应用筛选条件：将样本名和筛选关键词都转为小写进行匹配
                if filter_keyword and filter_keyword not in sample_name.lower():
                    continue
                plot_data.append({
                    "x": wavelength,
                    "y": intensity,
                    "name": sample_name
                })
        
        return plot_data if plot_data else None
    
    def _plot_original(self):
        plot_data = self._get_plot_data(use_processed=False)
        if not plot_data:
            messagebox.showwarning("警告", "没有符合条件的原始数据")
            return
            
        self.visualizer.plot_spectra(
            plot_data, 
            self.x_label.get(), 
            self.y_label.get(),
            title=self.plot_title.get()
        )
    
    def _plot_processed(self):
        plot_data = self._get_plot_data(use_processed=True)
        if not plot_data:
            messagebox.showwarning("警告", "没有符合条件的处理后数据")
            return
            
        self.visualizer.plot_spectra(
            plot_data, 
            self.x_label.get(), 
            self.y_label.get(),
            title=self.plot_title.get()
        )
    
    def _plot_analysis(self):
        if not self.analysis_results:
            messagebox.showwarning("警告", "请先执行分析")
            return
            
        method = self.analysis_results.get("方法", "")
        if method == "定量分析（单变量）":
            # 绘制校准曲线
            self.visualizer.plot_calibration_curve(
                self.analysis_results["浓度值"],
                self.analysis_results["强度值"],
                self.analysis_results["R²"],
                self.analysis_results["斜率"],
                self.analysis_results["截距"],
                title="定量分析校准曲线"
            )
        elif method == "定性分析（特征峰识别）":
            # 绘制带特征峰的光谱
            self.visualizer.plot_with_peaks(
                self.analysis_results["波长"],
                self.analysis_results["强度"],
                self.analysis_results["特征峰"],
                x_label=self.x_label.get(),
                y_label=self.y_label.get(),
                title=f"定性分析（{self.analysis_results['样本名']}）"
            )
        elif method == "主成分分析(PCA)":
            # 绘制PCA结果
            self.visualizer.plot_pca_results(
                self.analysis_results["主成分数据"],
                self.analysis_results["解释方差率"],
                title="PCA分析结果"
            )
    
    def _save_plot(self):
        if not hasattr(self.visualizer, 'fig') or self.visualizer.fig is None:
            messagebox.showwarning("警告", "请先绘制图表")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG图像", "*.png"), ("SVG矢量图", "*.svg"), ("所有文件", "*.*")]
        )
        
        if save_path:
            if self.visualizer.save_plot(save_path):
                self.status_var.set(f"图表已保存至: {save_path}")
            else:
                messagebox.showerror("错误", "保存图表失败")

if __name__ == "__main__":
    root = tk.Tk()
    app = SpectralAnalysisApp(root)
    root.mainloop()