import torch
import torch.nn as nn
from .utils import reset_params
# 这段代码定义了一个名为 TSK 的类，继承自 torch.nn.Module。
# 该类用于构建和训练基于T-S模糊神经网络（TSK模型）的模糊神经网络。
# TSK 类结合了前件部分（antecedent）和结论部分（cons），并通过前件部分计算每个规则的触发水平，
# 然后使用这些触发水平和特征来生成模型的最终预测。
# 同时提供了重新初始化参数、训练模型、进行预测和返回预测概率的方法。
class TSK(nn.Module):
    """

    父类: :code:`torch.nn.Module`

    该模块定义了TSK模型的结论部分，并将其与预定义的前件模块结合。该模块的输入是原始特征矩阵，输出是TSK模型的最终预测结果。

    :param int in_dim: 特征数量 :math:`D`。
    :param int out_dim: 输出维度 :math:`C`。
    :param int n_rule: 规则数量 :math:`R`，必须等于 :code:`Antecedent()` 的 :code:`n_rule`。
    :param torch.Module antecedent: 前件模块，其输出维度应等于规则数量 :math:`R`。
    :param int order: 0 或 1。TSK的阶数。如果为0，则使用零阶TSK；否则，使用一阶TSK。
    :param float eps: 一个常量，用于避免除零错误。
    :param torch.nn.Module precons: 如果为None，则使用原始特征作为结论输入；如果是一个Pytorch模块，则结论输入将是给定模块的输出。
    如果您希望使用我们在 `Models & Technique <../models.html#batch-normalization>`_ 中提到的BN技术，可以设置 :code:`precons=nn.BatchNorm1d(in_dim)`。

    """
    def __init__(self, in_dim, out_dim, n_rule, antecedent, order=1, eps=1e-8, precons=None):
        super(TSK, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_rule = n_rule
        self.antecedent = antecedent
        self.precons = precons

        self.order = order
        assert self.order == 0 or self.order == 1, "Order can only be 0 or 1阶数只能是0或1/"
        self.eps = eps

        self.__build_model__()

    def __build_model__(self):
        if self.order == 0:
            self.cons = nn.Linear(self.n_rule, self.out_dim, bias=True)# 零阶TSK的结论部分
        else:
            self.cons = nn.Linear((self.in_dim + 1) * self.n_rule, self.out_dim)# 一阶TSK的结论部分

    def reset_parameters(self):
        """
        重新初始化所有参数，包括结论部分和前件部分的参数。

        :return:
        """
        reset_params(self.antecedent)# 重新初始化前件模块的参数
        self.cons.reset_parameters()# 重新初始化结论部分的参数

        if self.precons is not None:
            self.precons.reset_parameters()# 如果有预处理层，重新初始化预处理层的参数

    def forward(self, X, get_frs=False):
        """

        :param torch.tensor X: 输入矩阵，大小为 :math:`[N, D]`，
            其中 :math:`N` 是样本数量。
        :param bool get_frs: 如果为True，则还会返回前件输出（前件部分的输出）。

        :return: 如果 :code:`get_frs=True`，则返回TSK输出 :math:`Y\in \mathbb{R}^{N,C}`
            和前件输出 :math:`U\in \mathbb{R}^{N,R}`。如果 :code:`get_frs=False`，
            则仅返回TSK输出 :math:`Y`。
        """
        frs = self.antecedent(X)# 获取前件的输出，即每个规则的触发水平

        if self.precons is not None:
            X = self.precons(X) # 如果有预处理层，应用预处理

        if self.order == 0:
            cons_input = frs # 零阶TSK的结论输入直接是触发水平
        else:
            X = X.unsqueeze(dim=1).expand([X.size(0), self.n_rule, X.size(1)])  # [n_batch, n_rule, in_dim]# 对于一阶TSK，扩展输入矩阵的维度以匹配规则数量
            X = X * frs.unsqueeze(dim=2)# 将特征与触发水平相乘
            X = X.view([X.size(0), -1]) # 将矩阵重塑为 [n_batch, (D+1)*R]
            cons_input = torch.cat([X, frs], dim=1)# 将特征和触发水平连接起来，形成结论输入

        output = self.cons(cons_input)# 使用结论部分进行前向传播以获得输出
        if get_frs:
            return output, frs# 如果需要触发水平，同时返回输出和触发水平
        return output # 否则仅返回输出



