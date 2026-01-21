import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
import os
from sklearn.decomposition import PCA
import networkx as nx
from sklearn.impute import SimpleImputer
import json
# 数据加载
SAVE_DIR = r'./results\figures'
adata = sc.read_h5ad(r'./data/processed_datasets/HeadNeck_Endocrine_processed.h5ad')


# 设置matplotlib参数以符合学术期刊要求
plt.rcParams['font.family'] = ['Arial']  # 设置统一字体
plt.rcParams['pdf.fonttype'] = 42  # 确保PDF中的字体可编辑
plt.rcParams['ps.fonttype'] = 42

def read_and_process_cancer_data(cancer_type, data_dir):
    """
    读取指定癌症类型的数据
    """
    adatas = []
    
    # 读取癌症组织数据 (counts01A.txt)
    tumor_file = os.path.join(data_dir, cancer_type, 'counts01A.txt')
    if os.path.exists(tumor_file):
        try:
            adata_tumor = sc.read_text(tumor_file)
            adata_tumor = adata_tumor.T  # 转置数据矩阵
            # 添加标签：癌症类型 + '_tumor'
            adata_tumor.obs['cancer_type'] = cancer_type
            adata_tumor.obs['label'] = f"{cancer_type}_tumor"
            adatas.append(adata_tumor)
            print(f"成功读取 {cancer_type} 的肿瘤组织数据")
        except Exception as e:
            print(f"读取或处理 {cancer_type} 的肿瘤组织数据时出错: {e}")
    else:
        print(f"警告: {cancer_type} 的肿瘤组织数据文件不存在")
    return adatas
def create_dataset(dataset_name, cancer_types, data_dir):
    """
    创建指定的数据集
    """
    all_adatas = []
    
    for cancer_type in cancer_types:
        cancer_adatas = read_and_process_cancer_data(cancer_type, data_dir)
        all_adatas.extend(cancer_adatas)
    
    if not all_adatas:
        print(f"警告: 数据集 {dataset_name} 没有有效的数据")
        return None
    
    # 合并所有数据
    combined_adata = sc.concat(all_adatas)  
    return combined_adata

# 设置数据目录和输出目录
data_dir = r"./data_all"


dataset_schemes = {
        "Digestive_System": {
            "cancer_types": ["ESCA", "STAD", "COAD", "READ", "LIHC", "CHOL", "PAAD"],
            "description": "消化系统癌"
        },
        "Thoracic_Respiratory": {
            "cancer_types": ["LUAD", "LUSC","MESO","THYM"],
            "description": "胸部与呼吸系统癌"
        },
        "Urogenital_System": {
            "cancer_types": ["BLCA", "PRAD","TGCT","OV", "KIRC", "KIRP", "KICH", "UCEC", "CESC","UCS"],
            "description": "泌尿生殖系统癌"
        },
        "HeadNeck_Endocrine": {
            "cancer_types": ["HNSC", "THCA", "UVM", "PCPG","SKCM"],
            "description": "头颈部与内分泌系统癌"
        }
    }

# 创建每个数据集
for dataset_name, scheme in dataset_schemes.items():
    print(f"\n正在加载数据集: {dataset_name} ({scheme['description']})")
    old_adata = create_dataset(
        dataset_name,
        scheme['cancer_types'], 
        data_dir, 
    )

# 检查 'wgcna_module' 列是否存在
if 'wgcna_module' not in adata.var.columns:
    print("Error: 'wgcna_module' column does not exist in adata.var")
    print("Available columns:", list(adata.var.columns))
else:
    # 读取分类后的基因
    module_genes = {}
    for module, genes in adata.var.groupby('wgcna_module').groups.items():
        module_genes[module] = list(genes)
    
    print(f"Found {len(module_genes)} gene modules")
    for module, genes in module_genes.items():
        print(f"Module {module}: {len(genes)} genes")
    

    # 创建数据副本以避免修改原始数据
    adata_preprocessed = adata.copy()
    
    # 检查零计数细胞
    # 处理adata_preprocessed.X可能是numpy数组或scipy稀疏矩阵的情况
    if hasattr(adata_preprocessed.X, 'A1'):
        # 对于scipy稀疏矩阵
        zero_count_cells = (adata_preprocessed.X.sum(axis=1) == 0).A1
    else:
        # 对于numpy数组
        zero_count_cells = (adata_preprocessed.X.sum(axis=1) == 0).ravel()
    
    if np.any(zero_count_cells):
        print(f"Found {zero_count_cells.sum()} cells with zero counts, will be filtered out")
        adata_preprocessed = adata_preprocessed[~zero_count_cells, :].copy()
    
    # 检查并处理NaN值
    if np.isnan(adata_preprocessed.X.data).any():
        print("NaN values detected in data, will be filled with mean values")
        # 对于稀疏矩阵，需要先转换为密集矩阵
        if hasattr(adata_preprocessed.X, 'toarray'):
            dense_X = adata_preprocessed.X.toarray()
        else:
            dense_X = adata_preprocessed.X
        
        # 使用SimpleImputer填充NaN值
        imputer = SimpleImputer(strategy='mean')
        dense_X_imputed = imputer.fit_transform(dense_X)
        
        # 将填充后的数据放回adata
        adata_preprocessed.X = dense_X_imputed
    
    # 归一化
    sc.pp.normalize_total(adata_preprocessed, target_sum=1e4)
    
    # 对数转换
    # 检查数据是否已经过对数转换
    min_val = np.min(adata_preprocessed.X)
    if min_val >= 0 and np.max(adata_preprocessed.X) < 20:
        # 如果数据范围表明它已经过对数转换，则跳过对数转换
        print("Data appears to be already log-transformed, skipping log1p step")
    else:
        sc.pp.log1p(adata_preprocessed)
    
    # 缩放
    sc.pp.scale(adata_preprocessed, max_value=10)
    
    # Preprocessing quality analysis and visualization
    print("\n===== Raw Data Quality Analysis =====")
    
    # Calculate cell quality metrics for raw data
    if hasattr(old_adata.X, 'toarray'):
        old_adata.obs['n_genes'] = (old_adata.X.toarray() > 0).sum(axis=1)
        old_adata.obs['total_counts'] = old_adata.X.toarray().sum(axis=1)
        # Calculate zero value percentage
        old_data_array = old_adata.X.toarray()
        zero_percentage = (old_data_array == 0).sum() / old_data_array.size * 100
    else:
        old_adata.obs['n_genes'] = (old_adata.X > 0).sum(axis=1)
        old_adata.obs['total_counts'] = old_adata.X.sum(axis=1)
        # Calculate zero value percentage
        zero_percentage = (old_adata.X == 0).sum() / (old_adata.X.shape[0] * old_adata.X.shape[1]) * 100
    
    print(f"Raw data zero value percentage: {zero_percentage:.2f}%")
    print(f"Raw data: {old_adata.n_obs} cells, {old_adata.n_vars} genes")
    
    # 准备原始数据用于后续对比图 - 不再单独保存图像
    sample_size = min(1000, old_adata.n_obs)
    sample_indices = np.random.choice(old_adata.n_obs, sample_size, replace=False)
    
    if hasattr(old_adata.X, 'toarray'):
        old_expr_data = old_adata[sample_indices, :100].X.toarray()
    else:
        old_expr_data = old_adata[sample_indices, :100].X
    
    # Plot zero value distribution heatmap (sample data)
    plt.figure(figsize=(7, 4))
    # Sample smaller subset for better performance
    small_sample = min(50, old_adata.n_obs)
    small_sample_indices = np.random.choice(old_adata.n_obs, small_sample, replace=False)
    
    if hasattr(old_adata.X, 'toarray'):
        small_sample_data = old_adata[small_sample_indices, :100].X.toarray()
    else:
        small_sample_data = old_adata[small_sample_indices, :100].X

    # 为保持代码功能完整性，仍保留零值比例的计算和基本输出
    if hasattr(old_adata.X, 'toarray'):
        old_zero_percentage = (old_adata.X.toarray() == 0).sum() / old_adata.X.toarray().size * 100
    else:
        old_zero_percentage = (old_adata.X == 0).sum() / (old_adata.X.shape[0] * old_adata.X.shape[1]) * 100
    
    if hasattr(adata_preprocessed.X, 'toarray'):
        processed_zero_percentage = (adata_preprocessed.X.toarray() == 0).sum() / adata_preprocessed.X.toarray().size * 100
    else:
        processed_zero_percentage = (adata_preprocessed.X == 0).sum() / (adata_preprocessed.X.shape[0] * adata_preprocessed.X.shape[1]) * 100
    
    print(f"零值比例: 原始数据 {old_zero_percentage:.2f}% -> 处理后数据 {processed_zero_percentage:.2f}%")
    print(f"零值减少: {(old_zero_percentage - processed_zero_percentage):.2f}%")
    
    # Post-processing quality metrics calculation and visualization
    print("\n===== Processed Data Quality Analysis =====")
    
    # Calculate each cell's gene count and total expression
    if hasattr(adata_preprocessed.X, 'toarray'):
        adata_preprocessed.obs['n_genes'] = (adata_preprocessed.X.toarray() > 0).sum(axis=1)
        adata_preprocessed.obs['total_counts'] = adata_preprocessed.X.toarray().sum(axis=1)
    else:
        adata_preprocessed.obs['n_genes'] = (adata_preprocessed.X > 0).sum(axis=1)
        adata_preprocessed.obs['total_counts'] = adata_preprocessed.X.sum(axis=1)
    
    print(f"Processed data: {adata_preprocessed.n_obs} cells, {adata_preprocessed.n_vars} genes")
    
    # 准备处理后数据用于后续对比图 - 不再单独保存图像
    if hasattr(adata_preprocessed.X, 'toarray'):
        expr_data = adata_preprocessed.X.toarray()[:, :100]
    else:
        expr_data = adata_preprocessed.X[:, :100]
    
    # Plot expression distribution comparison between raw and processed data
    plt.figure(figsize=(7, 3))
    
    # Raw data distribution
    plt.subplot(1, 2, 1)
    plt.hist(old_expr_data.flatten(), bins=100, alpha=0.7, color='#7293cb')
    plt.title('Raw Expression Distribution', fontsize=10)
    plt.xlabel('Expression Value', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    
    # Processed data distribution
    plt.subplot(1, 2, 2)
    plt.hist(expr_data.flatten(), bins=100, alpha=0.7, color='#e7969c')
    plt.title('Processed Expression Distribution', fontsize=10)
    plt.xlabel('Expression Value', fontsize=8)
    plt.ylabel('Frequency', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{dataset_name}_expression_comparison.png', dpi=600)
    plt.savefig(f'{SAVE_DIR}/{dataset_name}_expression_comparison.pdf', dpi=600)
    plt.close()
    print(f'Expression distribution comparison plot saved to {SAVE_DIR}/{dataset_name}_expression_comparison.png')
    print(f'矢量图表已保存至: {SAVE_DIR}/{dataset_name}_expression_comparison.pdf')
    
    
    # 添加：QC指标对比汇总
    print("\n===== Quality Control Metrics Summary =====")
    
    # 计算汇总指标
    metrics = {
        'Cells': [old_adata.n_obs, adata_preprocessed.n_obs],
        'Genes': [old_adata.n_vars, adata_preprocessed.n_vars],
        'Zero Percentage': [old_zero_percentage, processed_zero_percentage],
        'Mean Genes per Cell': [old_adata.obs['n_genes'].mean(), adata_preprocessed.obs['n_genes'].mean()],
        'Mean Total Counts': [old_adata.obs['total_counts'].mean(), adata_preprocessed.obs['total_counts'].mean()]
    }
    
    metrics_df = pd.DataFrame(metrics, index=['Raw Data', 'Processed Data'])
    print("\nQuality Control Metrics:")
    print(metrics_df)
    
    # 绘制雷达图对比QC指标
    categories = list(metrics.keys())
    N = len(categories)
    
    # 为雷达图准备数据
    values_raw = metrics_df.iloc[0].values
    values_processed = metrics_df.iloc[1].values
    
    # 归一化数据以便在雷达图上比较
    max_values = metrics_df.max().values
    values_raw_norm = values_raw / max_values
    values_processed_norm = values_processed / max_values
    
    # 角度设置
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 添加闭合点
    values_raw_norm = np.append(values_raw_norm, values_raw_norm[0])
    values_processed_norm = np.append(values_processed_norm, values_processed_norm[0])
    
    
    # 添加：数据处理前后的PCA对比
    print("\n===== PCA Comparison Before and After Processing =====")
    
    # 对原始数据进行PCA
    old_adata_sample = old_adata[np.random.choice(old_adata.n_obs, min(500, old_adata.n_obs), replace=False)]
    if hasattr(old_adata_sample.X, 'toarray'):
        old_data_pca = old_adata_sample.X.toarray()
    else:
        old_data_pca = old_adata_sample.X
    
    # 对处理后数据进行PCA
    adata_processed_sample = adata_preprocessed[np.random.choice(adata_preprocessed.n_obs, min(500, adata_preprocessed.n_obs), replace=False)]
    if hasattr(adata_processed_sample.X, 'toarray'):
        processed_data_pca = adata_processed_sample.X.toarray()
    else:
        processed_data_pca = adata_processed_sample.X
    
    # 应用PCA
    pca_old = PCA(n_components=2)
    pca_processed = PCA(n_components=2)
    
    old_pca_result = pca_old.fit_transform(old_data_pca)
    processed_pca_result = pca_processed.fit_transform(processed_data_pca)
    
    
    
    # Plot cell quality metrics comparison between raw and processed data
    plt.figure(figsize=(7, 3))
    
    # Raw data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='total_counts', y='n_genes', data=old_adata.obs, s=30, alpha=0.7, color='#7293cb')
    plt.title('Raw Cell Quality Metrics', fontsize=10)
    plt.xlabel('Total Counts per Cell', fontsize=8)
    plt.ylabel('Number of Genes per Cell', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    
    # Processed data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='total_counts', y='n_genes', data=adata_preprocessed.obs, s=30, alpha=0.7, color='#e7969c')
    plt.title('Processed Cell Quality Metrics', fontsize=10)
    plt.xlabel('Total Counts per Cell', fontsize=8)
    plt.ylabel('Number of Genes per Cell', fontsize=8)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{SAVE_DIR}/{dataset_name}_cell_quality_comparison.png', dpi=600)
    plt.savefig(f'{SAVE_DIR}/{dataset_name}_cell_quality_comparison.pdf', dpi=600)
    plt.close()
    print(f'Cell quality comparison plot saved to {SAVE_DIR}/{dataset_name}_cell_quality_comparison.png')
    print(f'矢量图表已保存至: {SAVE_DIR}/{dataset_name}_cell_quality_comparison.pdf')
    


    # 计算数据处理前后的信号噪声比变化
    if hasattr(old_adata.X, 'toarray'):
        old_data_array = old_adata.X.toarray()
    else:
        old_data_array = old_adata.X
    
    if hasattr(adata_preprocessed.X, 'toarray'):
        processed_data_array = adata_preprocessed.X.toarray()
    else:
        processed_data_array = adata_preprocessed.X
    
    old_snr = np.mean(old_data_array) / np.std(old_data_array) if np.std(old_data_array) > 0 else 0
    processed_snr = np.mean(processed_data_array) / np.std(processed_data_array) if np.std(processed_data_array) > 0 else 0
    
def load_json_file(file_path):
    """加载JSON文件，处理异常"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'")
    except json.JSONDecodeError:
        print(f"错误：'{file_path}' 不是有效的JSON格式")
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
    return None

input('请确认已运行完所有代码，按任意键继续...')
CAUSALTSK_JSON_PATH = r'./results/CausalTSK.json'
OTHER_MODEL_JSON_PATH = r'./results/other_model.json'

def load_json_file(file_path):
    """加载JSON文件，处理异常"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'")
    except json.JSONDecodeError:
        print(f"错误：'{file_path}' 不是有效的JSON格式")
    except Exception as e:
        print(f"读取文件时发生未知错误: {e}")
    return None

def get_filtered_data(data, target_models):
    """提取目标模型的数据"""
    return {model: data[model] for model in target_models if model in data}

def get_all_datasets_data(filtered_data):
    """获取所有数据集的数据"""
    dataset_names = list(next(iter(filtered_data.values())).keys())
    all_datasets_data = {}
    for dataset_name in dataset_names:
        all_datasets_data[dataset_name] = {
            model: filtered_data[model][dataset_name] 
            for model in filtered_data
        }
    return all_datasets_data, dataset_names

def calculate_avg_values(dataset_data):
    """计算每个模型各指标的平均值"""
    metrics = list(next(iter(dataset_data.values())).keys())
    model_avg_values = {}
    for model in dataset_data:
        model_avg_values[model] = [
            np.mean(dataset_data[model][metric]) for metric in metrics
        ]
    return model_avg_values, metrics


def plot_boxplot(all_datasets_data, dataset_names, metrics):
    """绘制分组箱线图（比较所有模型，排除TSK）"""
    os.makedirs(SAVE_DIR, exist_ok=True)

    datasets_to_plot = dataset_names[:]
    
    # 修正子图排版：2×2对称布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # 将2D数组展平为1D
    
    # 颜色设置
    colors = ['#7293cb', '#e7969c', '#fcae91', '#7f7f7f']
    palette = colors[:len(metrics)]
    
    for i, dataset_name in enumerate(datasets_to_plot):
        # 准备数据
        data_list, label_list, metric_list = [], [], []
        
        dataset_data = all_datasets_data[dataset_name]
        for metric in metrics:
            for model in dataset_data:
                data_list.extend(dataset_data[model][metric])
                label_list.extend([model] * len(dataset_data[model][metric]))
                metric_list.extend([metric] * len(dataset_data[model][metric]))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Value': data_list,
            'Model': label_list,
            'Metric': metric_list
        })
        
        # 绘制箱线图
        axes[i].set_title(dataset_name, fontsize=10)
        g = sns.boxplot(x='Model', y='Value', hue='Metric', data=df, 
                       palette=palette, boxprops=dict(alpha=0.7), ax=axes[i])
        sns.stripplot(x='Model', y='Value', hue='Metric', data=df, 
                     dodge=True, alpha=0.95, size=4, palette=palette, ax=axes[i])
        
        g.legend_.remove()  # 移除子图图例
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), ha='right')
    
    # 添加总图例
    handles, labels = axes[0].get_legend_handles_labels()
    unique_metrics = len(set(metric_list))
    fig.legend( handles[:unique_metrics],
                labels[:unique_metrics], 
                loc='center left', 
                bbox_to_anchor=(1.02, 0.5),
                ncol=1,
                fontsize=10
              )
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为图例留出空间
    
    # 保存图像
    plt.savefig(f'{SAVE_DIR}/model_comparison_boxplot.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{SAVE_DIR}/model_comparison_boxplot.pdf', dpi=600, bbox_inches='tight')
    plt.close()
    print(f'矢量图表已保存至: {SAVE_DIR}/model_comparison_boxplot.pdf')

def get_filtered_data(data, target_models):
    """提取目标模型的数据"""
    return {model: data[model] for model in target_models if model in data}


causaltsk_data = load_json_file(CAUSALTSK_JSON_PATH)
other_model_data = load_json_file(OTHER_MODEL_JSON_PATH)

RADAR_TARGET_MODELS = ['CTSK']
# 合并所有数据
combined_data = {}
combined_data['CTSK'] = causaltsk_data
dataset_names = list(next(iter(combined_data.values())).keys())
if other_model_data:
    combined_data.update(other_model_data)  # 添加其他模型
radar_data = get_filtered_data(combined_data, RADAR_TARGET_MODELS)
radar_all_datasets, dataset_names = get_all_datasets_data(radar_data)
_, metrics = calculate_avg_values(next(iter(radar_all_datasets.values())))
boxplot_models = [model for model in combined_data.keys() ]
boxplot_data = get_filtered_data(combined_data, boxplot_models)
boxplot_all_datasets, _ = get_all_datasets_data(boxplot_data)
plot_boxplot(boxplot_all_datasets, dataset_names, metrics)


output_dir = r'C:\Users\Administrator\Desktop\NC Code\result_cTSK'
#可视化消融对比

# ----------------------------------------------------
# 生成可视化对比图表
# ----------------------------------------------------
print("\n生成可视化对比图表...")

# 尝试加载已保存的汇总结果文件
summary_output_path = os.path.join(output_dir, 'CausalTSK_Modes_Summary.json')
if os.path.exists(summary_output_path):
    print(f"正在从文件加载结果: {summary_output_path}")
    with open(summary_output_path, 'r', encoding='utf-8') as json_file:
        summary_results = json.load(json_file)
    # 从加载的数据中提取模式名称
    if summary_results:
        first_dataset = next(iter(summary_results.values()))
        mode_names = list(first_dataset.keys())


# 提取数据进行可视化
dataset_names = list(summary_results.keys())

# 为每个模式定义不同的颜色 - 避免红绿对比，使用蓝橙色调
colors = {
    'causal_weight_reg : 1': '#1f77b4',  # 深蓝色
    'causal_weight_reg : 0.8': '#7293cb',  # 中蓝色
    'causal_weight_reg : 0.6': '#e7969c',  # 粉红色
    'causal_weight_reg : 0.4': '#fcae91',  # 浅橙色
    'causal_weight_reg : 0.2': '#fdbe85',  # 橙色
    'baseline': '#7f7f7f'  # 灰色
}

# 创建柱状图
plt.figure(figsize=(7, 4))

# 设置柱状图的宽度
bar_width = 0.15

# 计算每个数据集的中心位置
center_positions = list(range(len(dataset_names)))

# 为每个数据集和模式绘制柱状图
for i, dataset_name in enumerate(dataset_names):
    # 计算每个模式柱子的x位置 - 基于中心位置向两侧扩展
    offset = -bar_width * (len(mode_names) - 1) / 2
    x_positions = [center_positions[i] + offset + bar_width * j for j in range(len(mode_names))]
    
    # 获取每个模式的F1分数
    f1_scores = [summary_results[dataset_name][mode]['平均F1'] for mode in mode_names]
    
    # 绘制柱状图
    bars = plt.bar(x_positions, f1_scores, width=bar_width, label=dataset_name if i == 0 else "")
    

    for j, bar in enumerate(bars):
        # 设置柱子颜色
        bar.set_color(colors[mode_names[j]])

# Set up the plot
plt.xlabel('Dataset', fontsize=8)  # 字体大小符合要求
plt.ylabel('F1 Score', fontsize=8)
plt.title('F1 Score Comparison across Datasets and Training Modes', fontsize=10)
plt.xticks(center_positions, dataset_names, rotation=45, ha='right', fontsize=7)

# 添加图例 - 每个模式对应一种颜色
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[mode], lw=4, label=mode) for mode in mode_names]
plt.legend(handles=legend_elements, title='Training Mode', fontsize=7, loc='upper left')

# 设置y轴范围
plt.ylim(0.78, 1.0)  # 调整为合适的范围

# 添加网格线以便于比较
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 保存图表 - 保存为高分辨率PNG和可编辑的矢量PDF
chart_output_path_png = os.path.join(SAVE_DIR, 'model_comparison_f1_scores.png')
chart_output_path_pdf = os.path.join(SAVE_DIR, 'model_comparison_f1_scores.pdf')

plt.tight_layout()
plt.savefig(chart_output_path_png, dpi=600, bbox_inches='tight', format='png')  # 分辨率≥300dpi
plt.savefig(chart_output_path_pdf, format='pdf', bbox_inches='tight')  # 保存为可编辑的矢量文件
plt.close()

print(f"图表已保存至: {chart_output_path_png}")
print(f"矢量图表已保存至: {chart_output_path_pdf}")

'''
Causal-TSK 可视化复现代码
用于从保存的数据文件中重新生成所有图表
'''

import os
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import re
import sys

# 添加自定义模块路径
sys.path.append(os.getcwd())

# 导入必要的模型类
from tsk_model.tsk import TSK
from tsk_model.antecedent import AntecedentGMF

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ====================================================
# 1. 数据加载和模型重建函数
# ====================================================

def load_visualization_data(data_file_path):
    """
    加载保存的可视化数据
    """
    print(f"Loading visualization data from: {data_file_path}")
    with open(data_file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return data_dict

def reconstruct_model(data_dict):
    """
    根据保存的数据重建模型
    """
    print("Reconstructing TSK model...")
    
    # 首先重建antecedent
    antecedent = AntecedentGMF(
        in_dim=data_dict['x_data'].shape[1],
        n_rule=data_dict['n_rule'],
        init_center=None,  # 权重加载时会覆盖
        causal_factors=data_dict['causal_weights_prior'],
        learnable_causal=True,
    ).to(device)
    
    # 重建完整模型
    model = TSK(
        in_dim=data_dict['x_data'].shape[1], 
        out_dim=data_dict['n_class'],
        n_rule=data_dict['n_rule'],
        antecedent=antecedent,
        order=1,  # 默认使用1阶TSK
        precons=None
    ).to(device)
    
    # 加载模型权重
    model_weights_path = data_dict['model_weights_path']
    print(f"Loading model weights from: {model_weights_path}")
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    
    # 创建简化的wrapper用于预测
    class SimpleWrapper:
        def __init__(self, model):
            self.model = model
        
        def predict(self, x):
            with torch.no_grad():
                x_tensor = torch.tensor(x.astype('float32'), device=device)
                return self.model(x_tensor).cpu().numpy()
    
    wrapper = SimpleWrapper(model)
    
    return wrapper, model

# ====================================================
# 2. 规则解析函数
# ====================================================

def extract_fuzzy_rules(model, feature_names=None):
    """提取TSK模型的模糊规则"""
    antecedent = model.antecedent
    centers = antecedent.center.detach().cpu().numpy()
    sigmas = antecedent.sigma.detach().cpu().numpy()
    n_features = centers.shape[0]
    n_rules = centers.shape[1]

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(n_features)]

    rules = []
    for rule_idx in range(n_rules):
        rule_parts = []
        for feat_idx in range(n_features):
            center = centers[feat_idx, rule_idx]
            sigma = sigmas[feat_idx, rule_idx]
            if sigma < 1e-6:
                sigma = 1e-6
            rule_parts.append(f"{feature_names[feat_idx]} ~ N({center:.3f}, {sigma:.3f})")

        rule_str = f"IF {' AND '.join(rule_parts)}"
        rules.append(rule_str)
    return rules

def parse_rules_from_list(rule_text_list, feature_names):
    """
    解析规则文本，提取 μ 和 σ
    """
    parsed_rules = []
    gene_pattern = r'(\w+)\s*~.*?N\(([-+]?\d+\.\d+),\s*([-+]?\d+\.\d+)\)'

    for rule_id, rule_content in enumerate(rule_text_list, start=1):
        if_content = rule_content.replace('IF ', '', 1) 
        gene_matches = re.findall(gene_pattern, if_content)
        
        gene_params = {}
        for gene_name, mu_str, sigma_str in gene_matches:
            if gene_name in feature_names:
                gene_params[gene_name] = {
                    'mu': float(mu_str),
                    'sigma': float(sigma_str)
                }
        
        parsed_rules.append({
            'rule_id': rule_id,
            'genes': gene_params
        })
    
    print(f"Successfully parsed {len(parsed_rules)} TSK rules.")
    return parsed_rules

def analyze_tsk_rules(wrapper, x_train, feature_names=None):
    """
    分析TSK模型的规则
    """
    model = wrapper.model
    
    fuzzy_rules = extract_fuzzy_rules(model, feature_names)

    # 计算规则重要性
    with torch.no_grad():
        x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
        firing_strengths_train = model.antecedent(x_tensor).cpu().numpy()
        rule_importance = firing_strengths_train.mean(axis=0)

    return {
        '模糊规则': fuzzy_rules,
        '规则重要性': rule_importance,
        'firing_strengths_train': firing_strengths_train
    }

# ====================================================
# 3. 可视化函数
# ====================================================

def plot_all_visualizations(data_dict, file_prefix, output_dir):
    """
    绘制所有可视化图表
    """
    wrapper = data_dict['wrapper']
    model = wrapper.model
    
    # 数据提取
    x_data = data_dict['x_data']
    y_data = data_dict['y_data']
    feature_names = data_dict['feature_names']
    le = data_dict['le']
    causal_idx = data_dict['causal_idx']
    class_names = le.classes_.astype(str)
    
    # 固定颜色映射
    palette_classes = sns.color_palette("tab10", n_colors=len(class_names))
    color_map_classes = {name: color for name, color in zip(class_names, palette_classes)}
    hue_order_classes = class_names 
    
    # 分析规则
    analysis_results = analyze_tsk_rules(wrapper, data_dict['x_train_norm'], feature_names)
    rule_importance = analysis_results['规则重要性']
    
    # 设置模型为评估模式
    model.eval() 
    
    with torch.no_grad():
        y_pred_tensor = wrapper.predict(x_data)
        y_pred = np.argmax(y_pred_tensor, axis=1)
        x_tensor_all = torch.tensor(x_data.astype('float32'), device=device)
        frs_all = model.antecedent(x_tensor_all).cpu().numpy() 
        max_firing_rule = np.argmax(frs_all, axis=1)

    # t-SNE降维
    print("  -> Performing t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(x_data)
    df_tsne = pd.DataFrame(X_embedded, columns=['TSNE1', 'TSNE2'])
    
    # A1-A3: 降维图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # A1: 真实标签
    sns.scatterplot(x='TSNE1', y='TSNE2', hue=le.inverse_transform(y_data), 
                    palette=color_map_classes, hue_order=hue_order_classes, 
                    legend='full', ax=axes[0], data=df_tsne)
    axes[0].set_title('') 
    axes[0].legend(title='True Type')

    # A3: 规则覆盖
    print("  -> 绘制A3图: 最大触发规则分布...")
    
    # 确保规则编号在有效范围内

    # 转换为从1开始的规则编号用于显示
    display_rule_indices = max_firing_rule + 1
    
    # 创建规则标签
    rule_labels = [f'R{idx}' for idx in display_rule_indices]
    
    # 绘制A3图
    scatter = sns.scatterplot(x='TSNE1', y='TSNE2', hue=rule_labels, 
                    palette='viridis', legend='full', ax=axes[1], data=df_tsne)
    axes[1].set_title('') 
    
    # 获取图例中实际显示的规则
    legend = axes[1].get_legend()
    if legend is not None:
        legend_texts = [t.get_text() for t in legend.get_texts()]
        a3_displayed_rules = [int(text.replace('R', '')) for text in legend_texts]
        print(f"     A3图例中实际显示的规则: {a3_displayed_rules}")
    else:
        a3_displayed_rules = []
        print("     A3图没有显示图例")
    
    axes[1].legend(title='Max Rule Index', fontsize=8)

    # A2: 预测标签
    sns.scatterplot(x='TSNE1', y='TSNE2', hue=le.inverse_transform(y_pred), 
                    palette=color_map_classes, hue_order=hue_order_classes, 
                    legend='full', ax=axes[2], data=df_tsne)
    axes[2].set_title('') 
    axes[2].legend(title='Predicted Type')
    
    # 保存A3图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_TSNE_Rule_Coverage.png'), dpi=600)
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_TSNE_Rule_Coverage.pdf'), dpi=600)
    plt.close()

 #    # =========================================================================
    # 修改后的 B1: 规则-样本类型关联图 (X轴为样本类型，图例为规则，过滤低触发规则)
    # =========================================================================
    print("  -> 绘制修改后的规则-样本类型关联图...")
    n_rules = frs_all.shape[1] 
    n_classes = len(class_names)
    
    # 计算每个类别中每个规则的平均触发强度
    rule_class_mean_frs = np.zeros((n_classes, n_rules))  # 转置维度
    for i in range(n_classes):
        class_indices = (y_data == i) 
        rule_class_mean_frs[i, :] = np.mean(frs_all[class_indices, :], axis=0)

    # 过滤低触发强度的规则
    threshold = 0.01  # 设置阈值，平均触发强度低于此值的规则将被过滤
    rule_max_strength = np.max(rule_class_mean_frs, axis=0)  # 每个规则在所有类别中的最大触发强度
    significant_rules_mask = rule_max_strength > threshold
    significant_rule_indices = np.where(significant_rules_mask)[0]
    
    print(f"  总规则数: {n_rules}, 显著规则数: {len(significant_rule_indices)}")
    print(f"  被过滤的规则: {np.where(~significant_rules_mask)[0] + 1}")
    
    # 只保留显著规则的数据
    rule_class_mean_frs_filtered = rule_class_mean_frs[:, significant_rules_mask]
    significant_rule_names = [f'R{i+1}' for i in significant_rule_indices]
    
    # 创建DataFrame: 行=样本类型, 列=显著规则
    df_frs_filtered = pd.DataFrame(rule_class_mean_frs_filtered, 
                                   index=class_names, 
                                   columns=significant_rule_names)
    
    # 绘制堆叠条形图 (X轴为样本类型)
    plt.figure(figsize=(8, 6))
    
    # 为显著规则生成颜色
    n_significant_rules = len(significant_rule_names)
    rule_colors = plt.cm.tab20(np.linspace(0, 1, min(n_significant_rules, 20)))
    if n_significant_rules <= 10:
        rule_colors = sns.color_palette("pastel", n_significant_rules)
    else:
        rule_colors = sns.color_palette("husl", n_significant_rules)
    
    # 创建颜色映射字典
    color_dict = {rule: rule_colors[i] for i, rule in enumerate(significant_rule_names)}
    
    # 绘制堆叠条形图
    ax = df_frs_filtered.plot(kind='bar', stacked=True, figsize=(6, 4), 
                                color=[color_dict[rule] for rule in significant_rule_names],
                                ax=plt.gca())
    
    plt.title('')  # NC要求: 移除标题
    plt.ylabel('Average Firing Strength', fontsize=12)
    plt.xlabel('Sample Type', fontsize=12)
    
    # 优化图例 - 始终使用单列显示
    if n_significant_rules > 0:
        legend_cols = 1
        plt.legend(title='Rule Index', bbox_to_anchor=(1, 1), loc='upper left', 
                   fontsize=9, ncol=legend_cols)
        
        # 调整布局以适应图例
        legend_width = 0.15 
        plt.tight_layout(rect=[0, 0, 1-legend_width, 1])
    else:
        # 如果没有显著规则，移除图例
        ax.legend().remove()
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_Rule_Class_Frs.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_Rule_Class_Frs.pdf'), dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f"  -> 已保存修改后的B1图: {file_prefix}_Rule_Class_Frs.png")
    print(f"  -> 显示的显著规则: {significant_rule_names}")
    
    # 输出每个样本类型中触发强度最高的规则
    print("\n  每个样本类型的主要规则 (触发强度最高):")
    print("  " + "-" * 60)
    print(f"  {'样本类型':<15} {'主要规则':<10} {'触发强度':<12} {'次要规则':<20}")
    print("  " + "-" * 60)
    
    for class_idx, class_name in enumerate(class_names):
        # 获取当前类别中所有规则的触发强度
        class_intensities = rule_class_mean_frs_filtered[class_idx, :]
        
        if len(class_intensities) == 0:
            print(f"  {class_name:<15} {'无显著规则':<10} {'-':<12} {'-':<20}")
            continue
        
        # 找到触发强度最高的规则
        max_intensity_idx = np.argmax(class_intensities)
        max_intensity_rule = significant_rule_names[max_intensity_idx]
        max_intensity_value = class_intensities[max_intensity_idx]
        
        # 找到触发强度第二高的规则
        if len(class_intensities) > 1:
            sorted_indices = np.argsort(class_intensities)[::-1]
            second_max_rule = significant_rule_names[sorted_indices[1]]
            second_max_value = class_intensities[sorted_indices[1]]
            secondary_info = f"{second_max_rule}({second_max_value:.4f})"
        else:
            secondary_info = "-"
        
        print(f"  {class_name:<15} {max_intensity_rule:<10} {max_intensity_value:<12.4f} {secondary_info:<20}")
    print("  " + "-" * 60)
    
    # 输出每个显著规则的统计信息
    print("\n  显著规则触发强度统计:")
    print("  " + "-" * 70)
    print(f"  {'规则':<10} {'最高触发强度':<15} {'平均触发强度':<15} {'活跃样本类型数':<15}")
    print("  " + "-" * 70)
    
    for rule_idx, rule_name in enumerate(significant_rule_names):
        # 获取当前规则在所有样本类型中的触发强度
        rule_intensities = rule_class_mean_frs_filtered[:, rule_idx]
        
        # 计算统计信息
        max_intensity = np.max(rule_intensities)
        avg_intensity = np.mean(rule_intensities)
        active_classes = np.sum(rule_intensities > threshold/2)  # 半阈值以上算作活跃
        
        print(f"  {rule_name:<10} {max_intensity:<15.4f} {avg_intensity:<15.4f} {active_classes:<15}")
    print("  " + "-" * 70)

      # =========================================================================
    # 修改后的规则汇总图：只显示A3图例中实际显示的规则 + B1显著规则
    # =========================================================================
    print("  -> 绘制修改后的规则汇总图...")
    
    
    
    # 获取B1图中的显著规则（从0开始的索引）
    b1_significant_rules = significant_rule_indices
    
    # 合并A3图例中显示的规则和B1显著规则
    all_significant_rules = b1_significant_rules
    
    print(f"  -> B1图中显著规则: {b1_significant_rules + 1}")
    print(f"  -> 合并后的显著规则: {all_significant_rules + 1}")
    
    # 提取规则文本
    rule_text_list = extract_fuzzy_rules(model, feature_names)
    parsed_rules = parse_rules_from_list(rule_text_list, feature_names)
    
    # 调用修改后的规则汇总函数
    plot_rule_summary(
        parsed_rules=parsed_rules, 
        file_prefix=file_prefix, 
        output_dir=output_dir,
        significant_rule_indices=all_significant_rules,
        top_n_genes_per_rule=5
    )



    # C1: 规则和样本类型关联热图（B1图的补充说明）
    print("  -> Plotting rule-sample type association heatmap (supplement to B1)...")
    
    # 使用B1图中过滤后的规则数据
    if rule_class_mean_frs_filtered.shape[1] > 0:  # 确保有规则可显示
        plt.figure(figsize=(8, 6))
        
        # 绘制规则-样本类型关联热图
        sns.heatmap(rule_class_mean_frs_filtered, annot=False, cmap='YlOrRd', 
                    square=False, linewidths=.5, cbar_kws={'label': 'Mean Firing Strength'},
                    xticklabels=significant_rule_names, yticklabels=class_names)
        
        plt.title('')
        plt.xlabel('Rules')
        plt.ylabel('Sample Types')
        plt.xticks(fontsize=10, rotation=45, ha='right')
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_Rule_Sample_Association.png'), dpi=600)
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_Rule_Sample_Association.pdf'), dpi=600)
        plt.close()
        
        print(f"  -> 已保存规则-样本类型关联热图: {file_prefix}_Rule_Sample_Association.png")
        print(f"  -> 热图显示了{len(class_names)}个样本类型与{len(significant_rule_names)}个显著规则之间的关联强度")
        
        # 分析每个规则的主要关联样本类型
        print("\n  规则-样本类型主要关联:")
        print("  " + "-" * 60)
        print(f"  {'规则':<8} {'主要关联样本类型':<20} {'触发强度':<12} {'次要关联样本类型':<20}")
        print("  " + "-" * 60)
        
        for rule_idx, rule_name in enumerate(significant_rule_names):
            # 获取当前规则在所有样本类型中的触发强度
            rule_intensities = rule_class_mean_frs_filtered[:, rule_idx]
            
            # 找到触发强度最高的样本类型
            max_intensity_idx = np.argmax(rule_intensities)
            max_intensity_class = class_names[max_intensity_idx]
            max_intensity_value = rule_intensities[max_intensity_idx]
            
            # 找到触发强度第二高的样本类型
            if len(rule_intensities) > 1:
                sorted_indices = np.argsort(rule_intensities)[::-1]
                second_max_class = class_names[sorted_indices[1]]
                second_max_value = rule_intensities[sorted_indices[1]]
                secondary_info = f"{second_max_class}({second_max_value:.4f})"
            else:
                secondary_info = "-"
            
            print(f"  {rule_name:<8} {max_intensity_class:<20} {max_intensity_value:<12.4f} {secondary_info:<20}")
        print("  " + "-" * 60)
    else:
        print("  -> 跳过规则-样本类型关联热图（没有显著规则）")

def plot_rule_summary(parsed_rules, file_prefix, output_dir, 
                     significant_rule_indices=None, top_n_genes_per_rule=5):
    """
    将解析后的规则数据绘制到单个图表中，只显示显著规则
    :param parsed_rules: 包含每条规则参数（mu, sigma）的字典列表
    :param file_prefix: 文件前缀名
    :param output_dir: 图片保存目录
    :param significant_rule_indices: 显著规则的索引列表（从0开始）
    :param top_n_genes_per_rule: 每条规则显示前N个最重要的基因
    """
    if not parsed_rules:
        print('无有效规则可可视化。')
        return

    # 过滤规则，只显示显著规则
    if significant_rule_indices is not None:
        filtered_parsed_rules = []
        for rule in parsed_rules:
            # rule_id是从1开始的，significant_rule_indices是从0开始的规则索引
            if (rule['rule_id'] - 1) in significant_rule_indices:
                filtered_parsed_rules.append(rule)
        parsed_rules = filtered_parsed_rules
        print(f'  -> 过滤后保留 {len(parsed_rules)} 个显著规则')

    if not parsed_rules:
        print('无显著规则可可视化。')
        return

    print(f'  -> 正在生成规则单图合并可视化 (每规则Top {top_n_genes_per_rule} 基因)...')
    
    fig, ax = plt.subplots(figsize=(16, 8))  # 调整尺寸适应较少规则
    n_rules = len(parsed_rules)
    
    # 生成颜色和标记
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_rules, 20)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'P', 'X', 'd', '<', '>', '8', 'H'][:n_rules]
    
    all_mu_values = []
    rule_data_list = []
    
    for i, rule in enumerate(parsed_rules):
        rule_id = rule['rule_id']
        genes = rule['genes']
        
        if not genes:
            continue
        
        # 按|μ|排序并取前N个基因
        sorted_genes = sorted(genes.items(), key=lambda x: abs(x[1]['mu']), reverse=True)
        sorted_genes = sorted_genes[:top_n_genes_per_rule]
        
        # 提取基因数据
        gene_names = [g[0] for g in sorted_genes]
        mu_values = [g[1]['mu'] for g in sorted_genes]
        sigma_values = [g[1]['sigma'] for g in sorted_genes]
        
        # 输出规则的top5基因及其对应值
        print(f"\n规则 {rule_id} 的Top {top_n_genes_per_rule} 基因:")
        print("-" * 50)
        print(f"{'基因名称':<20} {'均值(μ)':<15} {'标准差(σ)':<15}")
        print("-" * 50)
        for gene_name, mu, sigma in zip(gene_names, mu_values, sigma_values):
            print(f"{gene_name:<20} {mu:<15.6f} {sigma:<15.6f}")
        print("-" * 50)
        
        rule_data_list.append({
            'rule_id': rule_id,
            'gene_names': gene_names,
            'mu_values': mu_values,
            'sigma_values': sigma_values
        })
        
        all_mu_values.extend(mu_values)
    
    # 确定统一的y轴范围
    if all_mu_values:
        y_min = min(all_mu_values) * 1.1 if min(all_mu_values) < 0 else min(all_mu_values) * 0.9
        y_max = max(all_mu_values) * 1.1 if max(all_mu_values) > 0 else max(all_mu_values) * 0.9
        if abs(y_max - y_min) < 0.5:
             y_max = max(y_max, 0.25)
             y_min = min(y_min, -0.25)
    else:
        y_min, y_max = -1, 1
    
    legend_handles = []
    
    # 绘制所有规则的基因数据
    for i, rule_data in enumerate(rule_data_list):
        rule_id = rule_data['rule_id']
        gene_names = rule_data['gene_names']
        mu_values = rule_data['mu_values']
        sigma_values = rule_data['sigma_values']
        
        # X轴偏移量：避免点重叠 - 增加偏移范围以分散点
        num_genes = len(mu_values)
        x_offsets = np.linspace(-0.45, 0.45, num_genes) if num_genes > 1 else [0]
        x_positions = [i + 1 + offset for offset in x_offsets]  # 重新编号从1开始
        
        # 绘制散点图
        scatter = ax.scatter(
            x=x_positions,
            y=mu_values,
            s=[sigma * 500 for sigma in sigma_values],
            c=[colors[i]] * num_genes,
            marker=markers[i % len(markers)],
            edgecolors='#222222',
            linewidth=0.7,
            alpha=0.8,
        )
        
        # 为图例添加一个代表性的点
        legend_handles.append(plt.scatter([], [], color=colors[i], marker=markers[i], 
                                         label=f'Rule {rule_id}', s=100))
        
        # 为每个点添加基因名称标签
        for j, (x, y, gene_name) in enumerate(zip(x_positions, mu_values, gene_names)):
            # 动态调整标签偏移方向，避免重叠
            offset_x = 10 if j % 2 == 0 else -10
            offset_y = 8 if j % 4 < 2 else -8
            
            ax.annotate(
                gene_name[:8] + ('...' if len(gene_name) > 8 else ''),
                (x, y),
                xytext=(offset_x, offset_y),
                textcoords='offset points',
                fontsize=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.8, edgecolor='none'),
                ha='center' if offset_x == 0 else ('left' if offset_x > 0 else 'right')
            )
    
    # 设置坐标轴和标题
    ax.set_title('')
    ax.set_xlabel('Rule ID', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Mean Value (μ) of Gene Expression (Normalized)', fontsize=12, fontweight='bold', labelpad=10)
    
    # 设置x轴刻度 - 使用实际的规则编号
    actual_rule_ids = [rule['rule_id'] for rule in parsed_rules]
    ax.set_xticks(range(1, n_rules + 1))
    ax.set_xticklabels([f'R{rule_id}' for rule_id in actual_rule_ids], 
                       rotation=45, ha='right', fontsize=10)
    
    # 设置y轴范围
    ax.set_ylim(y_min, y_max)
    ax.axhline(y=0, color='#666666', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # 添加图例
    if legend_handles:
        ax.legend(handles=legend_handles, title='Rules', ncol=1, fontsize=10, 
                  loc='center left', bbox_to_anchor=(1.02, 0.5), bbox_transform=ax.transAxes)
    
    # 添加大小说明 (sigma)
    from matplotlib.lines import Line2D
    size_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', 
               markersize=8, label='Small σ (Strict)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888', 
               markersize=20, label='Large σ (Tolerant)')
    ]
    # 移动到右上角并调整为横向排列以避免重合
    ax_legend = ax.inset_axes([0.72, 0.93, 0.25, 0.08])
    ax_legend.legend(handles=size_legend_elements, loc='center', ncol=2, fontsize=7, handletextpad=0.5)
    ax_legend.axis('off')
    
    plt.subplots_adjust(right=0.85)
    
    # 保存图像
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_TSK_Rules_Single_Plot.png'), 
                dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{file_prefix}_TSK_Rules_Single_Plot.pdf'), 
                dpi=600, bbox_inches='tight')
    plt.close()
# ====================================================
# 4. 主执行函数
# ====================================================

def reproduce_visualizations(data_file_path, output_dir, file_prefix=None):
    """
    主函数：复现所有可视化图表
    """
    print("=" * 60)
    print("Causal-TSK Visualization Reproduction")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data_dict = load_visualization_data(data_file_path)
    
    # 如果没有提供文件前缀，从数据文件名提取
    if file_prefix is None:
        file_prefix = os.path.basename(data_file_path).replace('_visual_data.pkl', '')
    
    # 重建模型
    wrapper, model = reconstruct_model(data_dict)
    
    # 更新数据字典，加入重建的模型对象
    data_dict['wrapper'] = wrapper
    data_dict['model'] = model
    
    print(f"\nReproducing visualizations for: {file_prefix}")
    print(f"Output directory: {output_dir}")
    
    # 生成基础可视化
    print("\n1. Generating basic visualizations...")
    plot_all_visualizations(data_dict, file_prefix, output_dir)
    

    print("\n" + "=" * 60)
    print("Visualization reproduction completed successfully!")
    print(f"All charts saved to: {output_dir}")
    print("=" * 60)

# ====================================================
# 5. 使用示例
# ====================================================


# 配置参数
DATA_FILE = r'./results/Digestive_visual_data.pkl'  # 修改为您的数据文件路径
OUTPUT_DIR = r'./\results\figures'  # 输出目录
FILE_PREFIX = 'Reproduced_Digestive'  # 可选：自定义文件前缀

# 执行可视化复现
reproduce_visualizations(DATA_FILE, OUTPUT_DIR, FILE_PREFIX)