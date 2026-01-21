'''
CTSK模型最优训练与可解释分析.py
版本：v3.0 (集成进度条 + 死循环修复 + 动态学习率/权重衰减搜索)
'''
import sys
import numpy as np
import os
import json
import warnings
import pickle
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
import scanpy as sc

# 贝叶斯优化模块
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

# 进度条模块
from tqdm import tqdm

# 确保路径正确
sys.path.append(os.getcwd())

# 导入自定义模块
try:
    from tsk_model.antecedent import AntecedentGMF, causal_antecedent_init_center
    from tsk_model.callbacks import EarlyStoppingACC
    from tsk_model.training import CausalWrapper
    from tsk_model.tsk import TSK
    from data_precess import run_causal_discovery_on_fold
except ImportError as e:
    print(f"导入自定义模块失败: {e}")
    sys.exit(1)

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


# 备用初始化函数
def causal_antecedent_init_center(X, y, causal_features, n_rule_per_feature=2, n_total_rules=20):
    from sklearn.cluster import KMeans
    if not causal_features:
        X_selected = X
    else:
        X_selected = X[:, causal_features]

    kmeans = KMeans(n_clusters=n_total_rules, random_state=42, n_init='auto')
    kmeans.fit(X_selected)
    centers = np.zeros((n_total_rules, X.shape[1]))
    if not causal_features:
        centers = kmeans.cluster_centers_
    else:
        for i, center in enumerate(kmeans.cluster_centers_):
            centers[i, causal_features] = center
    return torch.tensor(centers.T, dtype=torch.float32)


# ----------------------------------------------------
# 主执行代码
# ----------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tqdm.write(f"Device: {device}")
    if torch.cuda.is_available():
        torch.tensor([1.0], device=device)

    data_folder = './data/processed_datasets'

    # 【核心修复】严格过滤文件，防止读取副本导致无限循环
    if not os.path.exists(data_folder):
        tqdm.write(f"错误：文件夹不存在 {data_folder}")
        return

    all_files = os.listdir(data_folder)
    processed_files = [f for f in all_files if f.endswith('.h5ad') and 'copy' not in f and 'backup' not in f]

    tqdm.write(f"检测到 {len(processed_files)} 个数据集任务: {processed_files}")

    if len(processed_files) == 0:
        tqdm.write('请先运行数据预处理脚本生成 .h5ad 文件。')
        return

    results = {}
    output_dir = './results'
    image_output_dir = os.path.join(output_dir, 'PICTURE')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # --- [进度条层级 1]：数据集循环 ---
    dataset_pbar = tqdm(range(len(processed_files)), desc="总任务进度", position=0, dynamic_ncols=True)

    for t in dataset_pbar:
        current_file = processed_files[t]
        dataset_name_clean = change_files(current_file)

        dataset_pbar.set_description(f"正在处理数据集: {dataset_name_clean}")

        adata_file = os.path.join(data_folder, current_file)
        adata = sc.read_h5ad(adata_file)
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        ledge = adata.obs['label']
        if hasattr(adata.X, 'toarray'):
            umap_components = adata.X.toarray()
        else:
            umap_components = adata.X

        names = adata.var_names.tolist()
        le = LabelEncoder()
        y_encoded = le.fit_transform(ledge)
        n_class = len(le.classes_)

        result = {}
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        accuracies, precisions, recalls, f1s = [], [], [], []
        crosscheck = 0
        visualized = False

        tqdm.write(f"\n--- 开始处理: {current_file} (类别数: {n_class}) ---")

        # --- [进度条层级 2]：10折交叉验证 ---
        fold_pbar = tqdm(skf.split(umap_components, y_encoded), total=10, desc="CV Folds", position=1, leave=False,
                         dynamic_ncols=True)

        for train_index, test_index in fold_pbar:
            current_fold_num = crosscheck + 1
            fold_pbar.set_description(f"Fold {current_fold_num}/10")

            # 1. 划分数据
            x_train_raw, x_test_raw = umap_components[train_index], umap_components[test_index]
            y_train_fold, y_test_fold = y_encoded[train_index], y_encoded[test_index]
            y_train_labels = ledge.iloc[train_index].values

            # 2. 因果发现
            tqdm.write(f"   >>> [Fold {current_fold_num}] 启动因果发现...")
            try:
                fold_causal_weights, fold_causal_features_idx = run_causal_discovery_on_fold(
                    x_train_raw, y_train_labels, names, n_cores=4
                )
            except Exception as e:
                tqdm.write(f"   [警告] 因果发现异常: {e}，启用回退策略")
                fold_causal_features_idx = []
                fold_causal_weights = np.zeros(x_train_raw.shape[1])

            if len(fold_causal_features_idx) == 0:
                tqdm.write("   [信息] 未发现显著因果基因，使用高方差基因策略。")
                vars_ = np.var(x_train_raw, axis=0)
                fold_causal_features_idx = np.argsort(vars_)[-50:].tolist()
                for idx in fold_causal_features_idx:
                    fold_causal_weights[idx] = 1.0

            # 3. 标准化与验证集划分
            ss = StandardScaler()
            x_train = ss.fit_transform(x_train_raw)
            x_test = ss.transform(x_test_raw)
            x_train_sub, x_val, y_train_sub, y_val = train_test_split(x_train, y_train_fold, test_size=0.2,
                                                                      random_state=42)

            # 4. 贝叶斯优化 (更新后的搜索空间)
            search_spaces = [
                Integer(low=15, high=20, name='n_rule'),
                Real(low=0.0005, high=0.002, name='lr'),  # 学习率
                Real(low=1e-4, high=1e-3, name='weight_decay'),  # 权重衰减
                Real(low=0.2, high=0.8, name='ur_weight'),
                Real(low=0.2, high=0.8, name='causal_reg_weight')
            ]

            final_model_history = []
            order = 1
            N_CALLS = 50

            @use_named_args(search_spaces)
            def objective(**params):
                n_rule = int(params['n_rule'])
                current_lr = float(params['lr'])
                current_wd = float(params['weight_decay'])
                ur_weight = float(params['ur_weight'])
                causal_reg_weight = float(params['causal_reg_weight'])

                try:
                    init_center = causal_antecedent_init_center(
                        x_train_sub, y_train_sub, causal_features=fold_causal_features_idx,
                        n_rule_per_feature=2, n_total_rules=n_rule
                    )
                except:
                    from tsk_model.antecedent import antecedent_init_center
                    init_center = antecedent_init_center(x_train_sub, n_rule)

                gmf = AntecedentGMF(
                    in_dim=umap_components.shape[1], n_rule=n_rule, init_center=init_center,
                    causal_factors=fold_causal_weights, learnable_causal=True
                ).to(device)

                model = TSK(in_dim=umap_components.shape[1], out_dim=n_class, n_rule=n_rule,
                            antecedent=gmf, order=order).to(device)

                ante_param, causal_param, other_param = [], [], []
                for n, p in model.named_parameters():
                    if "center" in n or "sigma" in n:
                        ante_param.append(p)
                    elif "causal_layer" in n:
                        causal_param.append(p)
                    else:
                        other_param.append(p)

                # 动态优化器参数
                optimizer = AdamW([
                    {'params': ante_param, "weight_decay": 0},
                    {'params': causal_param, "weight_decay": 0},
                    {'params': other_param, "weight_decay": current_wd}
                ], lr=current_lr)

                temp_save_path = f"tmp_fold_{crosscheck}.pkl"
                EACC = EarlyStoppingACC(x_val, y_val, verbose=0, patience=20, save_path=temp_save_path)

                wrapper = CausalWrapper(
                    model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                    causal_weights=fold_causal_weights, causal_reg_weight=causal_reg_weight,
                    epochs=200, callbacks=[EACC], ur=ur_weight, ur_tau=1 / n_class,
                    device=device, causal_version=1
                )

                try:
                    wrapper.fit(x_train_sub, y_train_sub)
                except Exception:
                    pass

                try:
                    wrapper.load(temp_save_path)
                except:
                    pass

                y_pred_val = wrapper.predict(x_val).argmax(axis=1)
                metric = accuracy_score(y_val, y_pred_val) + f1_score(y_val, y_pred_val, average='weighted',
                                                                      zero_division=0)

                if crosscheck == 0:
                    final_model_history.extend(EACC.logs)

                return -metric

            # --- [进度条层级 3]：贝叶斯搜索 ---
            opt_pbar = tqdm(total=N_CALLS, desc="   贝叶斯寻优", position=2, leave=False, dynamic_ncols=True)

            def tqdm_callback(res):
                opt_pbar.update(1)

            res = gp_minimize(
                objective, search_spaces, n_calls=N_CALLS, random_state=42, verbose=False,
                callback=[tqdm_callback]
            )
            opt_pbar.close()

            # 获取所有最佳参数
            best_n_rule = int(res.x[0])
            best_lr = float(res.x[1])
            best_wd = float(res.x[2])
            best_ur_weight = float(res.x[3])
            best_causal_reg = float(res.x[4])

            tqdm.write(f"   [Fold {current_fold_num} 最佳参数] Rule={best_n_rule}, LR={best_lr:.5f}, WD={best_wd:.5f}")

            # 5. 最优模型重训
            x_train_full = np.vstack((x_train_sub, x_val))
            y_train_full = np.concatenate((y_train_sub, y_val))

            try:
                init_center_opt = causal_antecedent_init_center(
                    x_train_full, y_train_full, causal_features=fold_causal_features_idx,
                    n_rule_per_feature=2, n_total_rules=best_n_rule
                )
            except:
                from tsk_model.antecedent import antecedent_init_center
                init_center_opt = antecedent_init_center(x_train_full, best_n_rule)

            gmf_opt = AntecedentGMF(
                in_dim=umap_components.shape[1], n_rule=best_n_rule, init_center=init_center_opt,
                causal_factors=fold_causal_weights, learnable_causal=True
            ).to(device)

            model_opt = TSK(in_dim=umap_components.shape[1], out_dim=n_class, n_rule=best_n_rule,
                            antecedent=gmf_opt, order=order).to(device)

            ante_param, causal_param, other_param = [], [], []
            for n, p in model_opt.named_parameters():
                if "center" in n or "sigma" in n:
                    ante_param.append(p)
                elif "causal_layer" in n:
                    causal_param.append(p)
                else:
                    other_param.append(p)

            # 使用最佳 LR 和 WD
            optimizer_opt = AdamW([
                {'params': ante_param, "weight_decay": 0},
                {'params': causal_param, "weight_decay": 0},
                {'params': other_param, "weight_decay": best_wd}
            ], lr=best_lr)

            EACC_opt = EarlyStoppingACC(x_val, y_val, verbose=0, patience=30, save_path=f"opt_fold_{crosscheck}.pkl")

            wrapper_opt = CausalWrapper(
                model=model_opt, optimizer=optimizer_opt, criterion=nn.CrossEntropyLoss(),
                causal_weights=fold_causal_weights, causal_reg_weight=best_causal_reg,
                epochs=300, callbacks=[EACC_opt], ur=best_ur_weight, ur_tau=1 / n_class,
                device=device, causal_version=1
            )
            wrapper_opt.fit(x_train_full, y_train_full)

            try:
                wrapper_opt.load(f"opt_fold_{crosscheck}.pkl")
                if os.path.exists(f"opt_fold_{crosscheck}.pkl"):
                    os.remove(f"opt_fold_{crosscheck}.pkl")
            except:
                pass

            # 最终测试
            y_pred_test = wrapper_opt.predict(x_test).argmax(axis=1)
            accuracies.append(accuracy_score(y_test_fold, y_pred_test))
            precisions.append(precision_score(y_test_fold, y_pred_test, average='weighted', zero_division=0))
            recalls.append(recall_score(y_test_fold, y_pred_test, average='weighted', zero_division=0))
            f1s.append(f1_score(y_test_fold, y_pred_test, average='weighted', zero_division=0))

            # 保存可视化数据 (仅第1折)
            if crosscheck == 0 and not visualized:
                tqdm.write("   >>> 正在保存可视化数据...")
                current_file_prefix = change_files(processed_files[t])
                data_save_path = os.path.join(output_dir, f'{current_file_prefix}_visual_data.pkl')
                model_weights_path = os.path.join(output_dir, f'{current_file_prefix}_model_weights.pth')

                torch.save(wrapper_opt.model.state_dict(), model_weights_path)

                data_to_save = {
                    'x_data': umap_components,
                    'y_data': y_encoded,
                    'feature_names': names,
                    'le': le,
                    'causal_idx': fold_causal_features_idx,
                    'causal_weights_prior': fold_causal_weights,
                    'x_train_norm': x_train_full,
                    'loss_history': final_model_history,
                    'n_class': n_class,
                    'n_rule': best_n_rule,
                    'model_weights_path': model_weights_path,
                    'best_params': {
                        'n_rule': best_n_rule,
                        'lr': best_lr,
                        'wd': best_wd,
                        'ur': best_ur_weight,
                        'causal_reg': best_causal_reg
                    }
                }

                with open(data_save_path, 'wb') as f:
                    pickle.dump(data_to_save, f)

                visualized = True

            crosscheck += 1
            # End Fold Loop

        result['accuracy'] = accuracies
        result['precision'] = precisions
        result['recall'] = recalls
        result['f1'] = f1s
        results[f'{dataset_name_clean}'] = result
        # End Dataset Loop

    output_path = os.path.join(output_dir, 'CausalTSK.json')
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4, cls=NumpyEncoder)

    tqdm.write(f"\n全部任务完成！最终结果已保存至: {output_path}")


if __name__ == "__main__":
    main()