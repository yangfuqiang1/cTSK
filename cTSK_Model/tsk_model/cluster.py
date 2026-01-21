# 这个文件中的代码主要用于实现模糊C均值聚类算法，并将其与岭回归结合，用于构建TSK（Takagi-Sugeno-Kang）模糊系统。
# 通过这种方式，可以对数据进行模糊聚类，并进一步用于分类任务。

#导入库函数
#用于，数值计算，并更名为np
import numpy as np
#导入cdist函数，用于计算样本点之间的距离
from scipy.spatial.distance import cdist
#导入BaseEstimator和TransformerMixin，这些是scikit-learn库中的基类，用于实现自定义的估算器和转换器。
from sklearn.base import BaseEstimator, TransformerMixin
#导入check_array，用于验证输入的数组是否符合要求。
from sklearn.utils import check_array

#BaseFuzzyClustering类是scikit-learn库中的基类，用于实现自定义的模糊聚类器。
# 这是一个基础类，用于设置参数。它继承自object类。
# set_params方法允许通过关键字参数动态设置对象的属性。
class BaseFuzzyClustering(object):
    def set_params(self, **params):
        """
        设置属性。实现以适应scikit-learn的API。
        """
        for p, v in params.items():
            setattr(self, p, v)
        return self

# 根据输入样本数量N和特征数量D计算模糊指数m，如果m为"auto"，则根据公式计算，否则直接使用输入的值。
def __get_fuzzy_index__(m, N, D):
    """
       根据给定的参数计算模糊指数m。

       :param m: 模糊指数，可以是字符串'auto'，也可以是具体的浮点数或整数。
       :param N: 样本数量。
       :param D: 特征数量。
       :return: 计算得到的模糊指数m_。
    """
    if m == 'auto' and min(N, D - 1) >= 3:
        m_ = min(N, D - 1) / (min(N, D - 1) - 2)
    elif isinstance(m, float) or isinstance(m, int):
        m_ = float(m)
    else:
        m_ = 2
        print("Warning: auto set does not satisfied min(N, D - 1) >= 3, "
              "min(N, D - 1) = {}, setting m to 2".format(min(N, D - 1)))
    return m_

#归一化矩阵的列，确保每一列的和为1
def __normalize_column__(X):
    """
       对矩阵X的列进行归一化。
       :param X: 输入矩阵。
       :return: 归一化后的矩阵。
    """
    X = np.fmax(X, np.finfo(np.float64).eps)
    return X / np.sum(X, axis=0, keepdims=True)

#更新聚类中心cntr，隶属度矩阵u，以及损失loss，这是FCM算法的核心步骤。
def __fcm_update__(data, u_old, m, dist="euclidean"):
    """
       更新FCM算法中的聚类中心和隶属度矩阵。
       :param data: 输入数据矩阵。
       :param u_old: 上一次迭代的隶属度矩阵。
       :param m: 模糊指数。
       :param dist: 距离度量方法，默认为"euclidean"。
       :return: 新的聚类中心cntr，新的隶属度矩阵u，损失值loss，距离矩阵d。
    """
    u_old = __normalize_column__(u_old)
    u_old = np.fmax(u_old, np.finfo(np.float64).eps)
    um = u_old ** m

    cntr = np.dot(um, data) / um.sum(axis=1, keepdims=True)
    d = cdist(data, cntr, metric=dist).T
    d = np.fmax(d, np.finfo(np.float64).eps)

    loss = np.sum(um * d ** 2)
    u = __normalize_column__(d ** (2 / (1 - m)))
    return cntr, u, loss, d

#根据聚类中心cntr预测输入数据data的隶属度矩阵u。
def __fcm_predict__(data, cntr, m, dist="euclidean"):
    """
       预测输入数据在每个聚类上的隶属度。
       :param data: 输入数据矩阵。
       :param cntr: 聚类中心矩阵。
       :param m: 模糊指数。
       :param dist: 距离度量方法，默认为"euclidean"。
       :return: 隶属度矩阵u。
       """
    d = cdist(data, cntr, metric=dist).T
    d = np.fmax(d, np.finfo(np.float64).eps)
    u = __normalize_column__(d ** (2 / (1 - m)))
    return u

#将输入特征矩阵X和隶属度矩阵U转换为后续分析使用的后果输入矩阵Xp。根据TSK模型的阶数order决定输出格式。
def x2xp(X, U, order):
    """
    将矩阵X和U转换为后件输入矩阵X_p。
    :param numpy.array X: 形状为:math:`[N,D]`. 输入特征。
    :param numpy.array U: 形状为:math:`[N,R]`. 对应的隶属度矩阵。
    :param int order: 0或1. TSK模型的阶数。
    :return: 如果order=0，直接返回U，如果order=1，
    返回矩阵X_p，其形状为:math:`[N, (D+1)\times R]`。
    更多细节可以参考[2]。
    [2] Wang S, Chung K F L, Zhaohong D, 等. 基于ɛ-不敏感损失函数的鲁棒模糊聚类神经网络[J]. Applied Soft Computing, 2007, 7(2): 577-584.
    """

    assert order == 0 or order == 1, "Order can only be 0 or 1."
    R = U.shape[1]
    if order == 1:
        N = X.shape[0]
        mem = np.expand_dims(U, axis=1)
        X = np.expand_dims(np.concatenate((X, np.ones([N, 1])), axis=1), axis=2)
        X = np.repeat(X, repeats=R, axis=2)
        xp = X * mem
        xp = xp.reshape([N, -1])
        return xp
    else:
        return U

#计算输入特征矩阵X和隶属度矩阵U之间的方差，用于调整高斯隶属函数的标准差。
def compute_variance(X, U, V):
    """
       计算特征矩阵X和隶属度矩阵U的方差。

       :param numpy.array X: 形状为:math:`[N,D]`. 输入特征。
       :param numpy.array U: 形状为:math:`[N,R]`. 对应的隶属度矩阵。
       :param numpy.array V: 形状为:math:`[R,D]`. 聚类中心矩阵。
       :return: 方差矩阵variance，其形状为:math:`[R, D]`。
       """
    R = U.shape[0]
    D = X.shape[1]
    variance = np.zeros([R, D])
    for i in range(D):
        variance[:, i] = np.sum(
            U * ((X[:, i][:, np.newaxis] - V[:, i].T) ** 2).T, axis=1
        ) / np.sum(U, axis=1)
    return variance

# FuzzyCMeans类
class FuzzyCMeans(BaseFuzzyClustering, BaseEstimator, TransformerMixin):
    """
    模糊C均值（FCM）聚类算法。该实现基于`scikit-fuzzy`包。当构建TSK模糊系统时，通常使用模糊聚类算法来计算前件参数，
    然后使用最小二乘法（如岭回归）计算后件参数。

     模糊c均值 (FCM) 聚类算法 [1]. 该实现借鉴了
    `scikit-fuzzy <https://pythonhosted.org/scikit-fuzzy/overview.html>`_ 包。
    在构建TSK模糊系统时，通常使用模糊聚类算法来
    计算前件参数，然后可以使用最小二乘误差算法（如岭回归）来计算后件参数 [2]. 如何使用该类
     可以参考 `快速开始 <quick_start.html#training-with-fuzzy-clustering>`_。

    FCM的目标函数为：

    .. math::
        &J = \sum_{i=1}^{N}\sum_{j=1}^{C} U_{i,j}^m\|\mathbf{x}_i - \mathbf{v}_j\|_2^2\\
        &s.t. \sum_{j=1}^{C}\mu_{i,j} = 1, i = 1,...,N,

    :param int n_cluster: 聚类数量，等于TSK模型的规则数量 :math:`R`。
    :param float/str fuzzy_index: FCM算法的模糊指数，默认为`auto`。如果
        :code:`fuzzy_index=auto`，则模糊指数根据[3]计算为 :math:`\min(N, D-1) / (\min(N, D-1)-2)`
        （如果 :math:`\min(N, D-1)<3`，模糊指数将被设置为2）。否则使用给定的浮点值。
    :param float/str sigma_scale: 调整TSK前件部分高斯隶属函数实际标准差 :math:`\sigma` 的缩放参数 :math:`h`。
        如果 :code:`sigma_scale=auto`，:code:`sigma_scale` 将被设置为 :math:`\sqrt{D}`，其中 :math:`D` 是输入维度 [4]。
        否则使用给定的浮点值。
    :param str/np.array init: 隶属度网格矩阵 :math:`U` 的初始化策略。
        支持"random"或形状为:math:`[R, N]` 的numpy数组，其中 :math:`R` 是聚类/规则的数量，:math:`N` 是训练样本的数量。
        如果 :code:`init="random"`，隶属度网格矩阵将被随机初始化，否则使用给定的矩阵。
    :param int tol_iter: FCM算法的总迭代次数。
    :param float error: 在达到最大迭代次数之前，将停止迭代的最大误差。
    :param str dist: :func:`scipy.spatial.distance.cdist` 函数的距离类型，默认为"euclidean"。
        距离函数还可以是"braycurtis", "canberra", "chebyshev", "cityblock",
        "correlation", "cosine", "dice", "euclidean", "hamming", "jaccard", "jensenshannon", "kulsinski",
        "kulczynski1", "mahalanobis", "matching", "minkowski", "rogerstanimoto", "russellrao", "seuclidean",
        "sokalmichener", "sokalsneath", "sqeuclidean", "yule"。
    :param int verbose: 如果 > 0，在迭代过程中将显示FCM目标函数的损失值。
    :param int order: 0或1。决定构建零阶TSK还是第一阶TSK。
    """

    # FuzzyCMeans类继承自BaseFuzzyClustering，BaseEstimator和TransformerMixin，用于实现FCM算法
    #类的构造函数__init__定义了算法的参数，如聚类中心数量n_cluster，模糊指数fuzzy_index，标准差缩放参数sigma_scale等。
    # 初始化参数
    def __init__(self, n_cluster, fuzzy_index="auto", sigma_scale="auto",
                 init="random", tol_iter=100, error=1e-6, dist="euclidean",
                 verbose=0, order=1):
        self.n_cluster = n_cluster
        self.fuzzy_index = fuzzy_index
        self.sigma_scale = sigma_scale
        self.init = init
        self.tol_iter = tol_iter
        assert self.tol_iter > 1, "tol_iter must > 1."
        self.error = error
        self.dist = dist
        self.verbose = verbose
        self.order = order

    def get_params(self, deep=True):
        """
                获取模型参数。
                :param deep: 是否获取嵌套的对象参数。
                :return: 包含所有参数的字典。
                """
        return {
            "n_cluster": self.n_cluster,
            "fuzzy_index": self.fuzzy_index,
            "sigma_scale": self.sigma_scale,
            "init": self.init,
            "tol_iter": self.tol_iter,
            "error": self.error,
            "dist": self.dist,
            "verbose": self.verbose,
            "order": self.order,
        }
    #fit方法用于运行FCM算法。
    #它首先验证输入数组X，然后根据输入参数初始化聚类中心和隶属度矩阵。
    #通过迭代更新聚类中心和隶属度矩阵，直到隶属度矩阵的变化小于误差阈值error或达到最大迭代次数tol_iter。
    def fit(self, X, y=None):
        """
        运行FCM算法。

        :param numpy.array X: 输入数组，形状为:math:`[N, D]`，其中 :math:`N` 是训练样本的数量，
            :math:`D` 是特征数量。
        :param numpy.array y: 未使用。传递None。
        """
        check_array(X, ensure_2d=True)
        N, D = X.shape[0], X.shape[1]

        self.m_ = __get_fuzzy_index__(self.fuzzy_index, N, D)
        self.scale_ = np.sqrt(D) if self.sigma_scale == "auto" else self.sigma_scale
        self.n_features = D

        if self.init == "random":
            u = np.random.rand(self.n_cluster, N)
            u = __normalize_column__(u)
        elif isinstance(self.init, np.ndarray):
            u = __normalize_column__(self.init)
        else:
            raise ValueError("Unsupported init param, must be \"random\" or "
                             "numpy.ndarray of size [n_clusters, n_samples]")
        self.iter_cnt = 0
        cntr = None
        self.loss_hist = []
        for t in range(self.tol_iter):
            uold = u.copy()
            cntr, u, loss, d = __fcm_update__(X, uold, self.m_, self.dist)
            change = np.linalg.norm(uold - u, ord=2)
            self.loss_hist.append(loss)
            if self.verbose > 0:
                print('[FCM Iter {}] Loss: {:.4f}, change: {:.4f}'.format(t, loss, change))
            if change < self.error:
                break
            self.iter_cnt += 1
        self.fitted = True
        self.cluster_centers_ = cntr
        self.membership_degrees_ = u
        self.variance_ = compute_variance(X, self.membership_degrees_, self.cluster_centers_) * self.scale_

    #predict方法根据训练好的模型预测输入数据X的隶属度矩阵。
    def predict(self, X, y=None):
        """
        预测输入X在每个聚类上的隶属度。

        :param numpy.array X: 输入数组，形状为:math:`[N, D]`，其中 :math:`N` 是训练样本的数量，
            :math:`D` 是特征数量。
        :param numpy.array y: 未使用。传递None。
        :return: 返回隶属度矩阵 :math:`U`，形状为:math:`[N, R]`，其中 :math:`N`
            是X的样本数量，:math:`R` 是聚类/规则的数量。:math:`U_{i,j}`
            表示第 :math:`i` 个样本在第 :math:`r` 个聚类上的隶属度。
        """
        u = __fcm_predict__(X, self.cluster_centers_, self.m_, self.dist)
        return u.T

    #fit_transform方法首先调用fit方法训练模型，然后调用transform方法将输入数据转换为后果输入格式。
    def fit_transform(self, X, y=None, **fit_params):
        """
                先拟合模型，然后进行数据转换。
                :param X: 输入数据。
                :param y: 目标数据，未使用。
                :param fit_params: 其他拟合参数。
                :return: 转换后的数据。
        """
        self.fit(X, y)
        return self.transform(X, y)

    #transform方法根据训练好的模型和输入数据X计算隶属度矩阵U，然后使用x2xp函数将其转换为后果输入矩阵P。
    def transform(self, X, y=None):
        """
            计算隶属度矩阵 :math:`U`，并使用 :math:`X` 和 :math:`U` 通过函数 :func:`x2xp(x, u, order) <x2xp>`
            获取后件输入矩阵 :math:`P`。

            :param numpy.array X: 输入数组，形状为:math:`[N, D]`，其中 :math:`N` 是训练样本的数量，
            :math:`D` 是特征数量。
            :param numpy.array y: 未使用。传递None。
            :return: 返回后件输入 :math:`P`，其形状为:math:`[N, (D+1)\times R]`，其中
            :math:`N` 是测试样本的数量，:math:`D` 是特征数量，:math:`R` 是聚类/规则的数量。
        """
        d = -(np.expand_dims(X, axis=2) - np.expand_dims(self.cluster_centers_.T, axis=0)) ** 2 \
            / (2 * self.variance_.T + 1e-12)
        d = np.exp(np.sum(d, axis=1))
        d = np.fmax(d, 1e-16)
        u = d / np.sum(d, axis=1, keepdims=True)
        return x2xp(X, u, self.order)
