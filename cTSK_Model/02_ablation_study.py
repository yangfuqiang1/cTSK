'''
因果因子的发现函数 - 简化版本（三种模式对比）
嵌套交叉验证版 - 修复数据泄露
'''
import sys
import os
# 确保路径正确，以便导入自定义模块
sys.path.append(os.getcwd())
import numpy as np
import scanpy as sc
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from torch.optim import AdamW
# 导入自定义的梯度下降模块
from tsk_model.antecedent import AntecedentGMF, causal_antecedent_init_center
from tsk_model.callbacks import EarlyStoppingACC
from tsk_model.training import CausalWrapper
from tsk_model.tsk import TSK
# 引入因果发现函数
from data_precess import run_causal_discovery_on_fold
import os
import json
import warnings

warnings.filterwarnings("ignore")


# ----------------------------------------------------
# 辅助函数
# ----------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def change_files(filename):
    first_underscore_index = filename.find('_')
    if first_underscore_index != -1:
        word_before_underscore = filename[:first_underscore_index]
    else:
        word_before_underscore = filename[:filename.rfind('.')]
    return word_before_underscore


# ----------------------------------------------------
# 主执行代码
# ----------------------------------------------------

# Define random seed/定义随机种子以确保结果可复现
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    torch.tensor([1.0], device=device)

data_folder = './data/processed_datasets'
processed_files = [f for f in os.listdir(data_folder) if f.endswith('.h5ad')]
if len(processed_files) == 0:
    print('请先运行数据预处理脚本')
    exit()

# 定义和创建输出目录
output_dir = './results'
image_output_dir = os.path.join(output_dir, 'PICTURE')  # 图片新的保存目录

os.makedirs(output_dir, exist_ok=True)  # 确保 JSON 目录存在
os.makedirs(image_output_dir, exist_ok=True)  # 确保 PICTURE 目录存在
print(f"JSON results will be saved to: {output_dir}")
print(f"Image visualizations will be saved to: {image_output_dir}")

# 定义三种训练模式
TRAINING_MODES = {
    'causal_weight_reg : 1': {
        'description': '使用因果权重，因果正则化:1',
        'use_causal_weights': True,
        'causal_reg_weight': 1
    },
    'causal_weight_reg : 0.8': {
        'description': '使用因果权重，因果正则化:0.8',
        'use_causal_weights': True,
        'causal_reg_weight': 0.8
    },
    'causal_weight_reg : 0.6': {
        'description': '使用因果权重，因果正则化:0.6',
        'use_causal_weights': True,
        'causal_reg_weight': 0.6
    },
    'causal_weight_reg : 0.4': {
        'description': '使用因果权重，因果正则化:0.4',
        'use_causal_weights': True,
        'causal_reg_weight': 0.4
    },
    'causal_weight_reg : 0.2': {
        'description': '使用因果权重，因果正则化:0.2',
        'use_causal_weights': True,
        'causal_reg_weight': 0.2
    },
    'baseline': {
        'description': '基线：既不使用因果权重也不使用因果正则化',
        'use_causal_weights': False,
        'causal_reg_weight': 0
    }
}

# 固定的超参数（所有模式共享）
FIXED_HYPERPARAMS = {
    'n_rule': 18,
    'lr': 0.001,
    'weight_decay': 0.0005,
    'ur_weight': 0,
    'learnable_causal': True
}

# 存储所有模式的结果
all_results = {}

for t in range(len(processed_files)):
    adata_file = os.path.join(data_folder, processed_files[t])
    adata = sc.read_h5ad(adata_file)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    # 特征和标签
    ledge = adata.obs['label']
    # 使用全量原始特征（HVG Top 2000）
    x_raw_all = adata[:, :].X.toarray()
    names = adata.var_names.tolist()

    # 对标签进行编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(ledge)
    n_class = len(le.classes_)

    dataset_results = {}

    # 对每种模式进行训练
    for mode_name, mode_config in TRAINING_MODES.items():
        print(f"\n=== 处理数据集: {processed_files[t]} - 模式: {mode_name} ===")
        print(f"模式描述: {mode_config['description']}")

        result = {}
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        crosscheck = 0

        # 嵌套交叉验证循环
        for train_index, test_index in skf.split(x_raw_all, y_encoded):
            print(f'第 {crosscheck + 1} 次 10折交叉验证 - 模式: {mode_name}')

            # 1. 划分原始数据
            x_train_raw, x_test_raw = x_raw_all[train_index], x_raw_all[test_index]
            y_train, y_test = y_encoded[train_index], y_encoded[test_index]
            y_train_labels = ledge.iloc[train_index].values  # 需要原始标签字符串给WGCNA

            # 2. 根据模式决定是否运行因果发现
            use_causal_weights = mode_config['use_causal_weights']
            causal_reg_weight = mode_config['causal_reg_weight']

            current_causal_weights = None
            current_features_idx = []

            if use_causal_weights:
                # === 运行因果发现 (仅基于训练集) ===
                print("   正在运行因果发现...")
                fold_weights, fold_idx = run_causal_discovery_on_fold(
                    x_train_raw, y_train_labels, names, n_cores=4
                )

                # 保底机制
                if len(fold_idx) == 0:
                    print("   警告：未发现因果基因，使用高方差基因保底。")
                    vars_ = np.var(x_train_raw, axis=0)
                    fold_idx = np.argsort(vars_)[-50:].tolist()
                    for idx in fold_idx:
                        fold_weights[idx] = 1.0

                current_causal_weights = fold_weights
                current_features_idx = fold_idx

            else:
                # === Baseline 模式 ===
                # 不运行 FCI，而是使用基于方差的特征选择 (模拟普通TSK)
                # 并将权重设为均匀的 1.0
                print("   Baseline模式：使用高方差特征，无因果权重。")
                vars_ = np.var(x_train_raw, axis=0)
                # 选择 Top 50 高方差基因作为规则初始化的依据
                current_features_idx = np.argsort(vars_)[-50:].tolist()
                # 权重全为 1
                current_causal_weights = np.ones(x_train_raw.shape[1])

            # 3. 数据标准化 (Fit on Train, Transform on Test)
            ss = StandardScaler()
            x_train = ss.fit_transform(x_train_raw)
            x_test = ss.transform(x_test_raw)

            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

            order = 1

            # 使用固定超参数
            n_rule = FIXED_HYPERPARAMS['n_rule']
            lr = FIXED_HYPERPARAMS['lr']
            weight_decay = FIXED_HYPERPARAMS['weight_decay']
            ur_weight = FIXED_HYPERPARAMS['ur_weight']
            learnable_causal = FIXED_HYPERPARAMS['learnable_causal']

            # 构建模型
            init_center = causal_antecedent_init_center(
                x_train,
                y_train,
                causal_features=current_features_idx,  # 使用本折选出的特征(因果或高方差)
                n_rule_per_feature=2,
                n_total_rules=n_rule)

            gmf = AntecedentGMF(
                in_dim=x_raw_all.shape[1],
                n_rule=n_rule,
                init_center=init_center,
                causal_factors=current_causal_weights,
                learnable_causal=learnable_causal,
            ).to(device)

            model = TSK(in_dim=x_raw_all.shape[1], out_dim=n_class, n_rule=n_rule,
                        antecedent=gmf, order=order, precons=None).to(device)

            ante_param, causal_param, other_param = [], [], []
            for n, p in model.named_parameters():
                if "center" in n or "sigma" in n:
                    ante_param.append(p)
                elif "causal_layer" in n:
                    causal_param.append(p)
                else:
                    other_param.append(p)

            optimizer = (
                AdamW(
                    [
                        {'params': ante_param, "weight_decay": 0},
                        {'params': causal_param, "weight_decay": 0},
                        {'params': other_param, "weight_decay": weight_decay},
                    ],
                    lr=lr
                ))

            # 使用临时文件保存最佳模型
            temp_save_path = f"tmp_ablation.pkl"
            EACC = EarlyStoppingACC(x_val, y_val, verbose=0, patience=30, save_path=temp_save_path)

            wrapper = CausalWrapper(
                model=model,
                optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                causal_weights=current_causal_weights,
                causal_reg_weight=causal_reg_weight,  # 如果是Baseline，这里是0
                epochs=200,
                callbacks=[EACC],
                ur=ur_weight,
                ur_tau=1 / n_class,
                device=device,
                causal_version=3
            )
            wrapper.fit(x_train, y_train)

            # 加载早停时保存的最佳模型
            try:
                wrapper.load(temp_save_path)
            except:
                pass

            y_pred = wrapper.predict(x_test).argmax(axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            crosscheck += 1

        result['accuracy'] = accuracies
        result['precision'] = precisions
        result['recall'] = recalls
        result['f1'] = f1s
        result['description'] = mode_config['description']

        dataset_results[mode_name] = result

        # 打印该模式的平均性能
        print(f"\n模式 {mode_name} 平均性能:")
        print(f"准确率: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"F1分数: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    all_results[f'{change_files(processed_files[t])}'] = dataset_results

# ----------------------------------------------------
# 保存结果和性能比较
# ----------------------------------------------------

# 保存详细结果
detailed_output_path = os.path.join(output_dir, 'CausalTSK_Modes_Detailed.json')
with open(detailed_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(all_results, json_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

# 生成汇总结果
summary_results = {}
for dataset_name, modes in all_results.items():
    summary_results[dataset_name] = {}
    for mode_name, metrics in modes.items():
        summary_results[dataset_name][mode_name] = {
            '平均准确率': float(np.mean(metrics['accuracy'])),
            '平均精确率': float(np.mean(metrics['precision'])),
            '平均召回率': float(np.mean(metrics['recall'])),
            '平均F1': float(np.mean(metrics['f1'])),
            '描述': metrics['description']
        }

# 保存汇总结果
summary_output_path = os.path.join(output_dir, 'CausalTSK_Modes_Summary.json')
with open(summary_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(summary_results, json_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

print(f"\n详细结果保存至: {detailed_output_path}")
print(f"汇总结果保存至: {summary_output_path}")