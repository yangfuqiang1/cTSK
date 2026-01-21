'''
 模块显著性分析是否保留
 隐马尔可夫毯使用另一种方法存储,
 隐马尔可夫毯的获取方法:
 1.获取目标节点直接相连因果连接基因集合Y(A),A的直接邻居 adata.var['HMB']==2
 2.获取目标节点间接相连因果连接基因集合X(A),A的直接邻居的直接邻居 adata.var['HMB']==1
 3.计算Y(A)与X(A)的并集，得到隐马尔可夫毯
'''
import os
import pandas as pd
import numpy as np
import scanpy as sc
from PyWGCNA import WGCNA
import matplotlib.pyplot as plt
import matplotlib as mpl  # 引入mpl进行全局配置
from causallearn.search.ConstraintBased.FCI import fci
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from causallearn.utils.GraphUtils import GraphUtils
import multiprocessing
from sklearn.preprocessing import LabelEncoder
import logging
import pickle
import seaborn as sns

# -----------------------------------------------------------
# [NC 期刊绘图标准全局配置]
# -----------------------------------------------------------
# 1. 字体: 优先 Arial (NC首选) 或 Helvetica
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
# 2. 矢量化: 保证导出 PDF 时文字可编辑 (Type 42)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# 3. 分辨率与线条
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['axes.grid'] = False  # NC通常不推荐默认网格
# 4. Scanpy 设置
sc.settings.verbosity = 3
sc.set_figure_params(dpi=300, dpi_save=300, vector_friendly=True, frameon=False, facecolor='white')
# -----------------------------------------------------------

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 检查pyWGCNA是否成功导入的辅助函数
def check_pywgcna_installed():
    try:
        import PyWGCNA
        return True
    except ImportError:
        logger.warning("pyWGCNA库未安装。尝试WGCNA聚类功能时将跳过。")
        return False


PYWGCNA_AVAILABLE = check_pywgcna_installed()

# 因果关系类型权重映射
CAUSAL_WEIGHT_MAP = {
    -1: 1.0,  # A是B的直接原因
    2: 0.8,  # 存在直接连接
    1: 0.5,  # 存在潜在共同原因
    0: 0.0  # 无因果连接
}


def get_FCI(adata, module_str, dataset_name, p_value=None):
    """
    改进版 FCI 流程 (集成重叠分块与全局验证)
    """
    if not isinstance(module_str, str):
        # 兼容处理，如果是bytes或者其他类型转str
        module_str = str(module_str)

    adata1 = adata.copy()
    me_key = f'ME{module_str}'

    # 容错匹配 ME 列名
    if me_key not in adata1.obs.columns:
        possible_keys = [k for k in adata1.obs.columns if k.upper() == me_key.upper()]
        if possible_keys:
            me_key = possible_keys[0]
        else:
            # logger.warning(f"adata.obs 中不存在 '{me_key}' 列，跳过此模块")
            return [], []

    if 'wgcna_module' not in adata1.var.columns:
        raise KeyError("adata.var 中不存在 'wgcna_module' 列")

    # 筛选基因
    grey_genes_mask = adata1.var['wgcna_module'] == module_str
    adata1 = adata1[:, grey_genes_mask]

    # 如果模块内基因太少，跳过
    if adata1.n_vars < 2:
        return [], []

    x = adata1.X

    # 目标变量 Y
    Y = adata1.obs[me_key].values

    # 计算相关性并排序
    gene_names = adata1.var_names.tolist()
    correlations = []
    for i in range(x.shape[1]):
        corr, _ = pearsonr(x[:, i], Y)
        correlations.append((i, abs(corr)))
    correlations.sort(key=lambda x: x[1], reverse=True)

    # --- 改进 1: 重叠分块 (Overlapping Blocking) ---
    block_size = 10
    step = 10  # 50% 重叠

    blocks = []
    for i in range(0, len(correlations), step):
        current_slice = correlations[i: i + block_size]
        if len(current_slice) > 0:
            blocks.append(current_slice)
            if i + block_size >= len(correlations):
                break

    n_blocks = len(blocks)

    total_markov_blanket = {}  # {基因名: 权重}

    # 基础权重计算
    base_weights_map = {}
    if n_blocks > 0:
        for i in range(n_blocks):
            w = (n_blocks - i) / n_blocks
            if p_value is not None:
                w *= (1.0 / (max(0.0001, p_value) * 100))
            base_weights_map[i] = w
    else:
        return [], []

    le = LabelEncoder()
    # 确保label存在且为字符串
    y_labels = adata1.obs['label'].astype(str)
    y_encoded = le.fit_transform(y_labels)

    # ---> 第一阶段：局部块 FCI (Local Phase) <---
    # logger.info(f"模块 {module_str}: 开始第一阶段局部筛选 (共 {n_blocks} 个重叠块)...")

    for block_idx, current_block in enumerate(blocks):
        base_weight = base_weights_map.get(block_idx, 0.5)

        current_indices = [idx for idx, corr in current_block]
        gene_names_block = [gene_names[i] for i in current_indices]

        x_block = x[:, current_indices]
        x_combined = np.concatenate((x_block, y_encoded.reshape(-1, 1)), axis=1)

        nodes = np.append(gene_names_block, 'Y')

        try:
            g, edges = fci(
                x_combined,
                independence_test_method='fisherz',
                depth=2,
                max_path_length=2,
                node_names=nodes,
                alpha=0.01,
                verbose=False
            )

            graph = g.graph
            n_nodes = graph.shape[0]
            target_index = n_nodes - 1

            mb_indices = set()
            # 直接邻居
            for i in range(n_nodes - 1):
                if graph[target_index, i] != 0 or graph[i, target_index] != 0:
                    mb_indices.add(i)
            # 间接邻居
            direct_neighbors = list(mb_indices)
            for child in direct_neighbors:
                for i in range(n_nodes - 1):
                    if i != child and (graph[i, child] != 0 or graph[child, i] != 0):
                        mb_indices.add(i)

            for idx in mb_indices:
                gene_name = nodes[idx]
                w_fwd = CAUSAL_WEIGHT_MAP.get(graph[target_index, idx], 0.0)
                w_bwd = CAUSAL_WEIGHT_MAP.get(graph[idx, target_index], 0.0)
                struct_weight = max(w_fwd, w_bwd)
                final_weight = base_weight * struct_weight

                if gene_name not in total_markov_blanket or final_weight > total_markov_blanket[gene_name]:
                    total_markov_blanket[gene_name] = final_weight

        except Exception as e:
            continue

    # ---> 改进 2: 第二阶段全局精细化 (Global Refinement) <---
    sorted_candidates = sorted(total_markov_blanket.items(), key=lambda x: x[1], reverse=True)
    top_k = 20  # 限制数量防止卡死

    global_candidates = [gene for gene, weight in sorted_candidates[:top_k]]

    if len(global_candidates) > 1:
        # logger.info(f"模块 {module_str}: 开始第二阶段全局分析 (Top {len(global_candidates)} 基因)...")

        global_indices = [gene_names.index(g) for g in global_candidates]
        x_global = x[:, global_indices]
        x_combined_global = np.concatenate((x_global, y_encoded.reshape(-1, 1)), axis=1)
        nodes_global = np.append(global_candidates, 'Y')

        try:
            g_global, _ = fci(
                x_combined_global,
                independence_test_method='fisherz',
                depth=3,
                max_path_length=3,
                node_names=nodes_global,
                alpha=0.01,
                verbose=False
            )

            graph_g = g_global.graph
            target_idx_g = len(nodes_global) - 1

            for i, gene_name in enumerate(global_candidates):
                w_fwd = CAUSAL_WEIGHT_MAP.get(graph_g[target_idx_g, i], 0.0)
                w_bwd = CAUSAL_WEIGHT_MAP.get(graph_g[i, target_idx_g], 0.0)
                global_struct_weight = max(w_fwd, w_bwd)

                if global_struct_weight > 0:
                    bonus = 1.5
                    old_weight = total_markov_blanket[gene_name]
                    total_markov_blanket[gene_name] = old_weight * bonus
                    # logger.info(f"  基因 {gene_name} 通过全局验证，权重提升")

        except Exception as e:
            logger.warning(f"全局 FCI 分析失败: {e}")

    final_sorted = sorted(total_markov_blanket.items(), key=lambda x: x[1], reverse=True)
    final_genes = [g for g, w in final_sorted]
    final_weights = [w for g, w in final_sorted]

    return final_genes, final_weights


def wgcna_clustering(adata, min_module_size=50, power=6, module_size_ratio=0.1):
    """
    对基因表达数据进行WGCNA聚类分析
    """
    if not PYWGCNA_AVAILABLE:
        print("跳过WGCNA聚类，因为pyWGCNA库不可用。")
        return adata

    try:
        # print("开始WGCNA聚类分析...")
        expr_matrix = adata.X
        import scipy.sparse as sp
        if sp.issparse(expr_matrix):
            expr_matrix = expr_matrix.toarray()

        expr_matrix = np.nan_to_num(expr_matrix)

        # 简化的初始化与运行
        expr_df = pd.DataFrame(expr_matrix, index=adata.obs_names, columns=adata.var_names)

        # 确保唯一性
        if not expr_df.columns.is_unique:
            expr_df.columns = [f"{c}_{i}" for i, c in enumerate(expr_df.columns)]

        # 抑制 WGCNA 的详细输出
        wgcna = WGCNA(geneExp=expr_df, outputPath=None, save=False)

        # 设置参数
        if hasattr(wgcna, 'minModuleSize'): wgcna.minModuleSize = min_module_size
        if hasattr(wgcna, 'power'): wgcna.power = power
        if hasattr(wgcna, 'MEDissThres'): wgcna.MEDissThres = module_size_ratio

        wgcna.runWGCNA()

        # 获取模块
        module_colors = None
        if hasattr(wgcna, 'datExpr') and 'moduleColors' in wgcna.datExpr.var:
            module_colors = wgcna.datExpr.var['moduleColors'].tolist()
        elif hasattr(wgcna, 'moduleColors'):
            module_colors = wgcna.moduleColors

        if module_colors is None:
            module_colors = ['grey'] * adata.n_vars
        elif len(module_colors) != adata.n_vars:
            module_colors = module_colors[:adata.n_vars] + ['grey'] * max(0, adata.n_vars - len(module_colors))

        # 获取MEs
        if hasattr(wgcna, 'getMEs'):
            module_eigen_genes = wgcna.getMEs()
        elif hasattr(wgcna, 'datME'):
            module_eigen_genes = wgcna.datME
        else:
            module_eigen_genes = pd.DataFrame()

        for me_col in module_eigen_genes.columns:
            adata.obs[me_col] = module_eigen_genes[me_col].values

        adata.var['wgcna_module'] = module_colors
        # print(f"WGCNA聚类完成。")

    except Exception as e:
        print(f"WGCNA聚类过程中出错: {e}")
        adata.var['wgcna_module'] = 'grey'

    return adata


def process_color_module(args):
    """处理单个颜色模块的函数，用于并行处理"""
    color, adata, dataset_name, p_value = args
    # print(f'正在处理颜色模块: {color}')
    module_genes, module_fuzzies = get_FCI(adata, color, dataset_name=dataset_name, p_value=p_value)
    # print(f'  从{color}模块获取到{len(module_genes)}个基因')
    return module_genes, module_fuzzies


def run_causal_discovery_on_fold(x_train, y_train_labels, gene_names, n_cores=4):
    """
    【新增函数】专门用于在交叉验证的 Fold 内部运行因果发现。
    不生成图表，不保存文件，只返回因果权重。

    Args:
        x_train: 训练集特征矩阵 (numpy array)
        y_train_labels: 训练集标签 (原始字符串标签列表)
        gene_names: 基因名称列表
        n_cores: 并行核数

    Returns:
        fold_causal_weights: 对应 gene_names 的权重数组
        fold_causal_features_idx: 权重 > 0 的特征索引列表
    """

    # 1. 构建临时的 AnnData 对象
    adata_fold = sc.AnnData(X=x_train)
    adata_fold.var_names = pd.Index(gene_names)
    adata_fold.obs_names = pd.Index([f"Cell_{i}" for i in range(x_train.shape[0])])
    adata_fold.obs['label'] = y_train_labels

    # 2. 运行 WGCNA
    # 注意：min_module_size 可能需要根据 Fold 的大小适当调整，防止模块太小
    adata_fold = wgcna_clustering(adata_fold, min_module_size=30, power=6, module_size_ratio=0.2)

    # 3. 准备运行 FCI
    # 计算模块与 Label 的相关性/P值（简化版，仅用于筛选重要模块）
    me_columns = [col for col in adata_fold.obs.columns if col.startswith('ME')]
    color_to_pvalue = {}

    for me_col in me_columns:
        try:
            # 简单的两组间T检验，多组时取任意两组或使用方差分析
            # 这里为了速度，我们仅计算是否显著相关。
            # 如果Label多分类，可以使用Label Encoding后的相关性
            # 这里简化：计算 ME 与 Label(encoded) 的相关系数
            le = LabelEncoder()
            y_enc = le.fit_transform(adata_fold.obs['label'])
            corr, p_val = pearsonr(adata_fold.obs[me_col], y_enc)

            module_color = me_col[2:]
            color_to_pvalue[module_color] = p_val
        except:
            pass

    # 4. 获取所有模块颜色
    if 'wgcna_module' in adata_fold.var.columns:
        all_module_colors = [c for c in adata_fold.var['wgcna_module'].unique() if c and isinstance(c, str)]
    else:
        all_module_colors = []

    # 移除 grey 模块
    if 'grey' in all_module_colors:
        all_module_colors.remove('grey')

    # 5. 运行 FCI (并行)
    gene_list = []
    fuzzies_list = []

    if len(all_module_colors) > 0:
        if n_cores > 1:
            args_list = []
            for color in all_module_colors:
                p_value = color_to_pvalue.get(color, 0.05)  # 默认给0.05
                # 使用 copy 避免多进程冲突
                args_list.append((color, adata_fold.copy(), "fold_temp", p_value))

            # 使用 try-except 包裹多进程，防止环境问题
            try:
                with multiprocessing.Pool(processes=min(n_cores, len(all_module_colors))) as pool:
                    results = pool.map(process_color_module, args_list)
                for genes, fuzzies in results:
                    gene_list += genes
                    fuzzies_list += list(zip(genes, fuzzies))
            except Exception as e:
                print(f"Fold FCI 多进程失败，转为单进程: {e}")
                for color in all_module_colors:
                    genes, fuzzies = get_FCI(adata_fold, color, "fold_temp", color_to_pvalue.get(color, 0.05))
                    gene_list += genes
                    fuzzies_list += list(zip(genes, fuzzies))
        else:
            for color in all_module_colors:
                genes, fuzzies = get_FCI(adata_fold, color, "fold_temp", color_to_pvalue.get(color, 0.05))
                gene_list += genes
                fuzzies_list += list(zip(genes, fuzzies))

    # 6. 整合权重
    gene_to_fuzzy = {g: f for g, f in fuzzies_list}

    fold_causal_weights = np.zeros(len(gene_names))
    index_map = {name: i for i, name in enumerate(gene_names)}

    for g, w in gene_to_fuzzy.items():
        if g in index_map:
            fold_causal_weights[index_map[g]] = w

    fold_causal_features_idx = np.where(fold_causal_weights > 0)[0].tolist()

    return fold_causal_weights, fold_causal_features_idx


# 保留原有的 preprocess 用于第一次生成 h5ad 文件，但不进行因果分析
def preprocess(adata, perform_wgcna=False, dataset_name=None, use_multiprocessing=True, n_cores=None, **wgcna_params):
    # 标准预处理
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable == True]

    # 定义绘图目录（使用绝对路径）
    fixed_base_dir = './results/figures'
    plot_dir = os.path.join(fixed_base_dir, dataset_name)
    os.makedirs(plot_dir, exist_ok=True)
    sc.settings.figdir = plot_dir

    # 基础 PCA 和 UMAP (仅用于数据探索，不用于后续因果发现，因果发现在 Fold 中重新做)
    # print("\n--- V1: 正在计算 UMAP 降维 (预处理阶段) ---")
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)

    try:
        sc.pl.umap(adata, color='label', size=10,
                   title=f'{dataset_name} UMAP',
                   show=False,
                   save=f'_{dataset_name}_UMAP.png',
                   frameon=True,
                   legend_fontsize=10,
                   legend_loc='right margin',
                   )
    except Exception as e:
        pass

    # 注意：为了避免数据泄露，我们在此处运行全局 WGCNA 和 FCI。
    # WGCNA 和 FCI 将被移动到训练脚本的 Cross-Validation 循环中。
    # 这里只进行基础的 QC、Normalization 和 HVG 选择。

    # 初始化空的 causal_weights，防止后续代码报错
    adata.var['causal_weights'] = 0.0
    adata.var['selected'] = False

    return adata


def read_and_process_cancer_data(cancer_type, data_dir):
    adatas = []
    tumor_file = os.path.join(data_dir, cancer_type, 'counts01A.txt')
    if os.path.exists(tumor_file):
        try:
            adata_tumor = sc.read_text(tumor_file)
            adata_tumor = adata_tumor.T
            adata_tumor.obs['cancer_type'] = cancer_type
            adata_tumor.obs['label'] = f"{cancer_type}"
            adatas.append(adata_tumor)
            print(f"成功读取 {cancer_type}")
        except Exception as e:
            print(f"读取出错 {cancer_type}: {e}")
    return adatas


def create_dataset(dataset_name, cancer_types, data_dir, output_dir, perform_wgcna=False, **wgcna_params):
    all_adatas = []
    for cancer_type in cancer_types:
        cancer_adatas = read_and_process_cancer_data(cancer_type, data_dir)
        all_adatas.extend(cancer_adatas)

    if not all_adatas:
        return None

    combined_adata = sc.concat(all_adatas)
    # perform_wgcna 设置为 False，确保不在全局运行
    processed_adata = preprocess(combined_adata, perform_wgcna=False, dataset_name=dataset_name, **wgcna_params)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_processed.h5ad")
    processed_adata.write(output_file)
    return processed_adata


def main():
    data_dir = "./data_all"
    output_dir = "./data/processed_datasets"

    dataset_schemes = {
        "Digestive_System": {
            "cancer_types": ["ESCA", "STAD", "COAD", "READ", "LIHC", "CHOL", "PAAD"],
            "description": "消化系统癌"
        },
        "Thoracic_Respiratory": {
            "cancer_types": ["LUAD", "LUSC", "MESO", "THYM"],
            "description": "胸部与呼吸系统癌"
        },
        "Urogenital_System": {
            "cancer_types": ["BLCA", "PRAD", "TGCT", "OV", "KIRC", "KIRP", "KICH", "UCEC", "CESC", "UCS"],
            "description": "泌尿生殖系统癌"
        },
        "HeadNeck_Endocrine": {
            "cancer_types": ["HNSC", "THCA", "UVM", "PCPG", "SKCM"],
            "description": "头颈部与内分泌系统癌"
        }
    }

    # 即使这里设置 True，我们在 create_dataset 里也强制关掉了，
    # 确保因果发现只在训练脚本中进行。
    perform_wgcna = False
    wgcna_params = {
        'min_module_size': 50,
        'power': 6,
        'module_size_ratio': 0.2
    }

    for dataset_name, scheme in dataset_schemes.items():
        print(f"\n正在处理: {dataset_name}")
        create_dataset(
            dataset_name,
            scheme['cancer_types'],
            data_dir,
            output_dir,
            perform_wgcna=perform_wgcna,
            **wgcna_params
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()