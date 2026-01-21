#%%
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from torch.optim import AdamW
#定义其他模型
import xgboost as xgb
from sklearn.pipeline import Pipeline
from pytsk.cluster import FuzzyCMeans
from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
from pytsk.gradient_descent.callbacks import EarlyStoppingACC
from pytsk.gradient_descent.training import Wrapper
from pytsk.gradient_descent.tsk import TSK
from sklearn.linear_model import RidgeClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 导入FirstTSK模型和相关函数
from lib import FirstTSK
from lib import train_mini_batch, test
def change_files(filename):
    first_underscore_index = filename.find('_')

    # 提取第一个下划线之前的部分，去掉文件扩展名
    if first_underscore_index != -1:
        # 提取从文件名开始到第一个下划线前的部分
        word_before_underscore = filename[:first_underscore_index]
    else:
        # 如果没有下划线，去掉文件扩展名返回整个文件名
        word_before_underscore = filename[:filename.rfind('.')]
    return word_before_underscore

# Define random seed/定义随机种子以确保结果可复现
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
data_folder =r'./data/processed_datasets'
processed_files = [f for f in os.listdir(data_folder) if f.endswith('.h5ad')]
if len(processed_files) == 0:
    print('请先运行数据预处理脚本')
results = {}
results_TSK = {}
results_TSKgradient = {}
results_SVM = {}
results_DT = {}
results_RF = {}
results_TSKfcm = {}
results_XGB = {}
results_HDTSK = {}
results_all = {}
for t in range(len(processed_files)):
    adata_file = os.path.join(data_folder, processed_files[t])
    dataset_name = change_files(processed_files[t])
    adata = sc.read_h5ad(adata_file)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    label = adata.obs['label']
    # 根据文件类型处理数据
    X = adata.X
    # 将互信息分数与特征索引关联
    le = LabelEncoder()
    y_encoded = le.fit_transform(label).astype(int)
        # 根据任务类型选择合适的互信息计算方法
    # 使用特征索引选择adata中的特征
    umap_components = adata.X
    n_class = len(le.classes_)
    # Prepare dataset/准备数据集，生成一个包含1000个样本，20个特征，2个类别的二分类问题的数据集
    # split train-test/将数据集分为训练集和测试集，测试集占20%
    #%%
    result_SVM = {}
    result_DT = {}
    result_RF = {}
    result_TSKgradient = {}
    result_TSKfcm = {}
    result_XGB = {}
    result_HDTSK = {}

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    accuracies_TSKfcm = []
    precisions_TSKfcm = []
    recalls_TSKfcm = []
    f1s_TSKfcm = []
    accuracies_SVM = []
    precisions_SVM = []
    recalls_SVM = []
    f1s_SVM = []
    accuracies_DT = []
    precisions_DT = []
    recalls_DT = []
    f1s_DT = []
    accuracies_RF = []
    precisions_RF = []
    recalls_RF = []
    f1s_RF = []
    accuracies_TSKgradient = []
    precisions_TSKgradient = []
    recalls_TSKgradient = []
    f1s_TSKgradient = []
    accuracies_XGB = []
    precisions_XGB = []
    recalls_XGB = []
    f1s_XGB = []
    
    # HDTSK模型评估指标
    accuracies_HDTSK = []
    precisions_HDTSK = []
    recalls_HDTSK = []
    f1s_HDTSK = []

    for train_index, test_index in skf.split(umap_components, y_encoded):
        x_train, x_test = umap_components[train_index], umap_components[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    # Z-score/对训练集和测试集进行Z-score标准化
        ss = StandardScaler(with_mean=False)
        x_train = ss.fit_transform(x_train)# 用训练集数据拟合标准化参数，并对训练集进行标准化
        x_test = ss.transform(x_test)# 对测试集使用相同的标准化参数进行标准化

        print("Train on {} samples, test on {} samples, num. of features is {}, num. of class is {}".format(
            x_train.shape[0], x_test.shape[0], x_train.shape[1], n_class
        ))# 再次打印标准化后的数据集信息
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        #FirstTSK模型
        # 为FirstTSK准备数据：one-hot编码标签
        le_oh = LabelEncoder()
        y_train_encoded = le_oh.fit_transform(y_train).astype(int)
        y_test_encoded = le_oh.transform(y_test).astype(int)
        
        # one-hot编码
        num_classes = len(le_oh.classes_)
        y_train_onehot = torch.zeros((y_train_encoded.shape[0], num_classes))
        y_train_onehot[torch.arange(y_train_encoded.shape[0]), y_train_encoded] = 1
        y_test_onehot = torch.zeros((y_test_encoded.shape[0], num_classes))
        y_test_onehot[torch.arange(y_test_encoded.shape[0]), y_test_encoded] = 1
        
        # MinMaxScaler标准化到[0,1]区间
        min_max_scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_train)
        x_train_minmax = torch.Tensor(min_max_scaler.transform(x_train))
        x_test_minmax = torch.Tensor(min_max_scaler.transform(x_test))
        
        # 初始化FirstTSK模型
        num_fuzzy_set = 3
        num_fea = x_train_minmax.shape[1]
        myFirstTSK = FirstTSK(num_fea, num_classes, num_fuzzy_set, mf='Gaussian_DMF_sig')
        
        # 训练模型
        learning_rate = 0.001
        max_epoch = 100
        batch_size = 64
        train_mini_batch(x_train_minmax, myFirstTSK, y_train_onehot, learning_rate, max_epoch, batch_size=batch_size, optim_type='Adam')
        
        # 测试模型
        _, acc, precision, recall, f1 = test(x_test_minmax, myFirstTSK, y_test_onehot)
        accuracies_HDTSK.append(acc)
        precisions_HDTSK.append(precision)
        recalls_HDTSK.append(recall)
        f1s_HDTSK.append(f1)
        
        # TSK-gradient model
        # Define TSK model parameters
        n_rule = 30  # Num. of rules
        lr = 0.01  # learning rate
        weight_decay = 1e-8
        consbn = False
        order = 1

        # --------- Define antecedent ------------
        init_center = antecedent_init_center(x_train, y_train, n_rule=n_rule)
        gmf = nn.Sequential(
            AntecedentGMF(in_dim=X.shape[1], n_rule=n_rule, high_dim=True, init_center=init_center),
            nn.LayerNorm(n_rule),
            nn.ReLU()
        )
        # set high_dim=True is highly recommended.

        # --------- Define full TSK model ------------
        model = TSK(in_dim=X.shape[1], out_dim=n_class, n_rule=n_rule, antecedent=gmf, order=order, precons=None)
        # ----------------- optimizer ----------------------------
        ante_param, other_param = [], []
        for n, p in model.named_parameters():
            if "center" in n or "sigma" in n:
                ante_param.append(p)
            else:
                other_param.append(p)
        optimizer = AdamW(
            [{'params': ante_param, "weight_decay": 0},
             {'params': other_param, "weight_decay": weight_decay}, ],
            lr=lr
        )
        # ----------------- split 10% data for earlystopping -----------------
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        # ----------------- define the earlystopping callback -----------------
        EACC = EarlyStoppingACC(x_val, y_val, verbose=1, patience=20, save_path="tmp.pkl")
        wrapper = Wrapper(model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                          epochs=300, callbacks=[EACC])
        wrapper.fit(x_train, y_train)
        wrapper.load("tmp.pkl")
        TSK_gradient_pred = wrapper.predict(x_test).argmax(axis=1)
        accuracies_TSKgradient = accuracies_TSKgradient + [accuracy_score(y_test, TSK_gradient_pred)]
        precisions_TSKgradient = precisions_TSKgradient + [precision_score(y_test, TSK_gradient_pred, average='macro', zero_division=0)]
        recalls_TSKgradient = recalls_TSKgradient + [recall_score(y_test, TSK_gradient_pred, average='macro')]
        f1s_TSKgradient = f1s_TSKgradient + [f1_score(y_test, TSK_gradient_pred, average='macro')]

        #CSV model
        Svm = svm.SVC(kernel='rbf', C =1.0,random_state=42)
        Svm.fit(x_train, y_train)
        CSV_test_pred = Svm.predict(x_test)
        accuracies_SVM = accuracies_SVM +[accuracy_score(y_test, CSV_test_pred)]
        precisions_SVM = precisions_SVM + [precision_score(y_test, CSV_test_pred,average='macro')]
        recalls_SVM = recalls_SVM + [recall_score(y_test, CSV_test_pred, average='macro')]
        f1s_SVM = f1s_SVM + [f1_score(y_test, CSV_test_pred, average='macro')]

        #DT model
        Dt = DecisionTreeClassifier(random_state=42)
        Dt.fit(x_train, y_train)
        DT_test_pred = Dt.predict(x_test)
        accuracies_DT = accuracies_DT + [accuracy_score(y_test, DT_test_pred)]
        precisions_DT = precisions_DT + [precision_score(y_test, DT_test_pred, average='macro')]
        recalls_DT = recalls_DT + [recall_score(y_test, DT_test_pred, average='macro')]
        f1s_DT = f1s_DT + [f1_score(y_test, DT_test_pred, average='macro')]

        #RF model
        Rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        Rf.fit(x_train, y_train)
        RF_test_pred = Rf.predict(x_test)
        accuracies_RF = accuracies_RF + [accuracy_score(y_test, RF_test_pred)]
        precisions_RF = precisions_RF + [precision_score(y_test, RF_test_pred, average='macro')]
        recalls_RF = recalls_RF + [recall_score(y_test, RF_test_pred, average='macro')]
        f1s_RF = f1s_RF + [f1_score(y_test, RF_test_pred, average='macro')]

        #XGB model
        XGB = xgb.XGBClassifier(random_state=42, n_estimators=2)
        XGB.fit(x_train, y_train)
        XGB_test_pred = XGB.predict(x_test)
        accuracies_XGB = accuracies_XGB + [accuracy_score(y_test, XGB_test_pred)]
        precisions_XGB = precisions_XGB + [precision_score(y_test, XGB_test_pred, average='macro')]
        recalls_XGB = recalls_XGB + [recall_score(y_test, XGB_test_pred, average='macro')]
        f1s_XGB = f1s_XGB + [f1_score(y_test, XGB_test_pred, average='macro')]

        #TSK-fcm model
        FCM_n_rule = 20
        FCM_model = Pipeline(
            steps=[
                ("GaussianAntecedent", FuzzyCMeans(FCM_n_rule, sigma_scale="auto", fuzzy_index="auto")),
                ("Consequent", RidgeClassifier())
            ]
        )
        FCM_model.fit(x_train, y_train)
        TSKfcm_pred = FCM_model.predict(x_test)
        accuracies_TSKfcm = accuracies_TSKfcm + [accuracy_score(y_test, TSKfcm_pred)]
        precisions_TSKfcm = precisions_TSKfcm + [precision_score(y_test, TSKfcm_pred, average='macro',zero_division=0)]
        recalls_TSKfcm = recalls_TSKfcm + [recall_score(y_test, TSKfcm_pred, average='macro')]
        f1s_TSKfcm = f1s_TSKfcm + [f1_score(y_test, TSKfcm_pred, average='macro')]
    
    
    
    result_RF['accuracy'] = accuracies_RF
    result_RF['precision'] = precisions_RF
    result_RF['recall'] = recalls_RF
    result_RF['f1'] = f1s_RF
    results_RF[f'{change_files(processed_files[t])}'] = result_RF
    results_all['RF'] = results_RF

    result_XGB['accuracy'] = accuracies_XGB
    result_XGB['precision'] = precisions_XGB
    result_XGB['recall'] = recalls_XGB
    result_XGB['f1'] = f1s_XGB
    results_XGB[f'{change_files(processed_files[t])}'] = result_XGB
    results_all['XGB'] = results_XGB
    
    result_TSKfcm['accuracy'] = accuracies_TSKfcm
    result_TSKfcm['precision'] = precisions_TSKfcm
    result_TSKfcm['recall'] = recalls_TSKfcm
    result_TSKfcm['f1'] = f1s_TSKfcm
    results_TSKfcm[f'{change_files(processed_files[t])}']= result_TSKfcm
    results_all['TSK-FCM'] = results_TSKfcm

    result_SVM['accuracy'] = accuracies_SVM
    result_SVM['precision'] = precisions_SVM
    result_SVM['recall'] = recalls_SVM
    result_SVM['f1'] = f1s_SVM
    results_SVM[f'{change_files(processed_files[t])}']= result_SVM
    results_all['SVM'] = results_SVM

    result_DT['accuracy'] = accuracies_DT
    result_DT['precision'] = precisions_DT
    result_DT['recall'] = recalls_DT
    result_DT['f1'] = f1s_DT
    results_DT[f'{change_files(processed_files[t])}'] = result_DT
    results_all['DT'] = results_DT

    result_TSKgradient['accuracy'] = accuracies_TSKgradient
    result_TSKgradient['precision'] = precisions_TSKgradient
    result_TSKgradient['recall'] = recalls_TSKgradient
    result_TSKgradient['f1'] = f1s_TSKgradient
    results_TSKgradient[f'{change_files(processed_files[t])}'] = result_TSKgradient
    results_all['TSK-gradient'] = results_TSKgradient
    
    # 保存HDTSK模型结果
    result_HDTSK['accuracy'] = accuracies_HDTSK
    result_HDTSK['precision'] = precisions_HDTSK
    result_HDTSK['recall'] = recalls_HDTSK
    result_HDTSK['f1'] = f1s_HDTSK
    results_HDTSK[f'{change_files(processed_files[t])}'] = result_HDTSK
    results_all['HDTSK'] = results_HDTSK

with open(r'./results/other_model.json', 'w', encoding='utf-8') as json_file:
    json.dump(results_all, json_file, ensure_ascii=False, indent=4)
