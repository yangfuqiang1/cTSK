import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.cluster import KMeans
from torch.nn import functional

from .utils import check_tensor
import importlib.util

# For illustrative purposes.
package_name = 'pandas'

spec = importlib.util.find_spec(package_name)
if spec is None:
    print(package_name +" is not installed")


def antecedent_init_center(X, y=None, n_rule=2, method="kmean", engine="sklearn", n_init=20):
    """

    This function run KMeans clustering to obtain the :code:`init_center` for :func:`AntecedentGMF() <AntecedentGMF>`.

    Examples
    --------
    >>> init_center = antecedent_init_center(X, n_rule=10, method="kmean", n_init=20)
    >>> antecedent = AntecedentGMF(X.shape[1], n_rule=10, init_center=init_center)


    :param numpy.array X: Feature matrix with the size of :math:`[N,D]`, where :math:`N` is the
        number of samples, :math:`D` is the number of features.
    :param numpy.array y: None, not used.
    :param int n_rule: Number of rules :math:`R`. This function will run a KMeans clustering to
        obtain :math:`R` cluster centers as the initial antecedent center for TSK modeling.
    :param str method: Current version only support "kmean".
    :param str engine: "sklearn" or "faiss". If "sklearn", then the :code:`sklearn.cluster.KMeans()`
        function will be used, otherwise the :code:`faiss.Kmeans()` will be used. Faiss provide a
        faster KMeans clustering algorithm, "faiss" is recommended for large datasets.
    :param int n_init: Number of initialization of the KMeans algorithm. Same as the parameter
        :code:`n_init` in :code:`sklearn.cluster.KMeans()` and the parameter :code:`nredo` in
        :code:`faiss.Kmeans()`.
    """
    def faiss_cluster_center(X, y=None, n_rule=2, n_init=20):
        import faiss
        km = faiss.Kmeans(d=X.shape[1], k=n_rule, nredo=n_init)
        km.train(np.ascontiguousarray(X.astype("float32")))
        centers = km.centroids.T
        return centers

    if method == "kmean":
        if engine == "faiss":
            package_name = "faiss"
            spec = importlib.util.find_spec(package_name)
            if spec is not None:
                center = faiss_cluster_center(X=X, y=y, n_rule=n_rule)
                return center
            else:
                print("Package " + package_name + " is not installed, running scikit-learn KMeans...")
        km = KMeans(n_clusters=n_rule, n_init=n_init)
        km.fit(X)
        return km.cluster_centers_.T


class LearnableCausalWeights(nn.Module):
    """
    可学习的因果权重层
    """
    def __init__(self, n_features, init_weights=None, learnable=True):
        super(LearnableCausalWeights, self).__init__()
        self.learnable = learnable
        if init_weights is None:
            self.weights = nn.Parameter(torch.ones(n_features), requires_grad=learnable)
        else:
            self.weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32), 
                                      requires_grad=learnable)
        
    def forward(self, x):
        return x * self.weights.unsqueeze(0)
    
    def get_weights(self):
        """获取当前的因果权重"""
        return self.weights.detach().cpu().numpy()


class Antecedent(nn.Module):
    def forward(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def reset_parameters(self):
        raise NotImplementedError


class AntecedentGMF(Antecedent):
    """

    Parent: :code:`torch.nn.Module`

    The antecedent part with Gaussian membership function. Input: data, output the corresponding
    firing levels of each rule. The firing level :math:`f_r(\mathbf{x})` of the
    :math:`r`-th rule are computed by:

    .. math::
        &\mu_{r,d}(x_d) = \exp(-\frac{(x_d - m_{r,d})^2}{2\sigma_{r,d}^2}),\\
        &f_{r}(\mathbf{x})=\prod_{d=1}^{D}\mu_{r,d}(x_d),\\
        &\overline{f}_r(\mathbf{x}) = \frac{f_{r}(\mathbf{x})}{\sum_{i=1}^R f_{i}(\mathbf{x})}.


    :param int in_dim: Number of features :math:`D` of the input.
    :param int n_rule: Number of rules :math:`R` of the TSK model.
    :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`,
        HTSK is used. Otherwise the original defuzzification is used. More details can be found at [1].
        TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True` is highly
         recommended for any-dimensional problems.
    :param numpy.array init_center: Initial center of the Gaussian membership function with
        the size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
        :code:`init_center` as the obtained centers. You can simply run
        :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
        to obtain the center.
    :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
    :param float eps: A constant to avoid the division zero error.
    :param numpy.array causal_factors: 因果权重数组，大小为 :math:`[D]`
    :param bool learnable_causal: 因果权重是否可学习
    """
    def __init__(self, in_dim, n_rule, high_dim=False, init_center=None, init_sigma=1., eps=1e-8, 
                 causal_factors=None, learnable_causal=True):
        super(AntecedentGMF, self).__init__()
        self.in_dim = in_dim
        self.n_rule = n_rule
        self.high_dim = high_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_sigma = init_sigma
        self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps
        self.learnable_causal = learnable_causal
        if causal_factors is None:
            self.causal_factors = torch.ones(in_dim)
        # 使用可学习的因果权重层
        self.causal_layer = LearnableCausalWeights(in_dim, causal_factors, learnable_causal)
        
        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))
        self.sigma = nn.Parameter(torch.zeros(size=(self.in_dim, self.n_rule)))

        self.reset_parameters()

    def init(self, center, sigma):
        """

        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function with the
            size of :math:`[D,R]`. A common way is to run a KMeans clustering and set
            :code:`init_center` as the obtained centers. You can simply run
            :func:`pytsk.gradient_descent.antecedent.antecedent_init_center <antecedent_init_center>`
            to obtain the center.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.
        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.

        :return:
        """
        init.constant_(self.sigma, self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        # 应用因果权重
        weighted_X = self.causal_layer(X)
        
        weighted_dist = -(weighted_X.unsqueeze(dim=2) - self.center) ** 2 * (0.5 / (self.sigma ** 2 + self.eps))
        frs = self.zr_op(weighted_dist, dim=1)
        frs = functional.softmax(frs, dim=1)
        return frs
    
    def get_causal_weights(self):
        """获取当前的因果权重"""
        return self.causal_layer.get_weights()


class AntecedentShareGMF(Antecedent):
    def __init__(self, in_dim, n_mf=2, high_dim=False, init_center=None, init_sigma=1., eps=1e-8):
        """
        The antecedent part with Gaussian membership function, rules will share the membership
        functions on each feature [2]. The number of rules will be :math:`M^D`, where :math:`M`
        is :code:`n_mf`, :math:`D` is the number of features (:code:`in_dim`).

        :param int in_dim: Number of features :math:`D` of the input.
        :param int n_mf: Number of membership functions :math:`M` of each feature.
        :param bool high_dim: Whether to use the HTSK defuzzification. If :code:`high_dim=True`,
            HTSK is used. Otherwise the original defuzzification is used. More details can be found
            at [1]. TSK model tends to fail on high-dimensional problems, so set :code:`high_dim=True`
            is highly recommended for any-dimensional problems.
        :param numpy.array init_center: Initial center of the Gaussian membership function with
            the size of :math:`[D,M]`.
        :param float init_sigma: Initial :math:`\sigma` of the Gaussian membership function.
        :param float eps: A constant to avoid the division zero error.

        [1] `Cui Y, Wu D, Xu Y. Curse of dimensionality for tsk fuzzy neural networks:
        Explanation and solutions[C]//2021 International Joint Conference on Neural Networks
        (IJCNN). IEEE, 2021: 1-8. <https://arxiv.org/pdf/2102.04271.pdf>`_
        """
        super(AntecedentShareGMF, self).__init__()

        self.in_dim = in_dim
        self.n_mf = n_mf
        self.n_rule = self.n_mf ** self.in_dim
        self.high_dim = high_dim

        self.init_center = check_tensor(init_center, torch.float32) if init_center is not None else None
        self.init_sigma = init_sigma
        self.zr_op = torch.mean if high_dim else torch.sum
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        self.center = nn.Parameter(torch.normal(0, 1, size=(self.in_dim, self.n_mf)))
        self.sigma = nn.Parameter(torch.ones(size=(self.in_dim, self.n_mf)) * self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)

        self.rule_index = list(itertools.product(*[range(self.n_mf) for _ in range(self.in_dim)]))

        self.reset_parameters()

    def init(self, center, sigma):
        """
        Change the value of :code:`init_center` and :code:`init_sigma`.

        :param numpy.array center: Initial center of the Gaussian membership function
            with the size of :math:`[D,M]`.
        :param float sigma: Initial :math:`\sigma` of the Gaussian membership function.
        """
        center = check_tensor(center, torch.float32)
        self.init_center = center
        self.init_sigma = sigma

        self.reset_parameters()

    def reset_parameters(self):
        """
        Re-initialize all parameters.
        :return:
        """
        init.constant_(self.sigma, self.init_sigma)

        if self.init_center is not None:
            self.center.data[...] = torch.FloatTensor(self.init_center)
        else:
            init.normal_(self.center, 0, 1)

    def forward(self, X):
        """
        Pytorch模块的前向传播方法。
        :param torch.tensor X: 输入张量，尺寸为[N, D]，其中N是样本数量，D是输入维度
        :return: 规则触发强度矩阵U，尺寸为[N, R]，其中R=M^D
        """
        zrs = []
        for r in range(self.n_rule):
            mf_index = torch.tensor(self.rule_index[r], device=X.device, dtype=torch.long).unsqueeze(-1)
            center, sigma = torch.gather(self.center, 1, mf_index), torch.gather(self.sigma, 1, mf_index)

            zr = -0.5 * (X.unsqueeze(2) - center) ** 2 / (sigma ** 2 + self.eps)
            zr = self.zr_op(zr, dim=1)
            zrs.append(zr)
        zrs = torch.cat(zrs, dim=1)
        frs = functional.softmax(zrs, dim=1)
        return frs


# 新增的因果相关函数
def causal_antecedent_init_center(X, y, causal_features, n_rule_per_feature=2, n_total_rules=10):
    """
    基于因果重要特征生成规则中心
    
    :param numpy.array X: 特征矩阵
    :param numpy.array y: 标签
    :param list causal_features: 因果重要特征的索引列表
    :param int n_rule_per_feature: 每个重要特征的规则数量
    :param int n_total_rules: 总规则数量
    :return: 规则中心矩阵，大小为 [D, R]
    """
    centers = []
    
    # 为每个因果重要特征创建规则
    for feat_idx in causal_features:
        values = X[:, feat_idx]
        feat_centers = np.linspace(np.min(values), np.max(values), n_rule_per_feature)
        for center in feat_centers:
            # 创建规则：只在重要特征上有中心，其他特征用全局均值
            rule_center = np.mean(X, axis=0).copy()
            rule_center[feat_idx] = center
            centers.append(rule_center)
    
    # 补充规则
    if len(centers) < n_total_rules:
        n_additional = n_total_rules - len(centers)
        km = KMeans(n_clusters=n_additional)
        km.fit(X)
        centers.extend(km.cluster_centers_)
    
    # 如果规则数量超过要求，随机选择
    if len(centers) > n_total_rules:
        indices = np.random.choice(len(centers), n_total_rules, replace=False)
        centers = [centers[i] for i in indices]
    
    return np.array(centers).T