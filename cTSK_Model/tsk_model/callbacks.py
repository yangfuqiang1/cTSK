from sklearn.metrics import accuracy_score


class Callback:
    """
       类似于Keras中的回调类，我们的包提供了一个简化的回调版本，
       允许用户在训练过程中监控指标。
       我们强烈建议用户自定义回调，这里我们提供了两个示例，
       :func:`EvaluateAcc <EvaluateAcc>` 和 :func:`EarlyStoppingACC <EarlyStoppingACC>`。
    """
    def on_batch_begin(self, wrapper):
        pass

    def on_batch_end(self, wrapper):
        pass

    def on_epoch_begin(self, wrapper):
        pass

    def on_epoch_end(self, wrapper):
        pass


class EvaluateAcc(Callback):
    """

       在训练过程中评估准确率。

       :param numpy.array X: 特征矩阵，大小为 :math:`[N, D]`，其中 :math:`N` 是样本数，:math:`D` 是特征数。
       :param numpy.array y: 标签矩阵，大小为 :math:`[N, 1]`。
       """
    def __init__(self, X, y, verbose=0):
        super(EvaluateAcc, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.logs = []

    def on_epoch_end(self, wrapper):
        # 在每个epoch结束时计算并记录准确率
        cur_log = {}
        y_pred = wrapper.predict(self.X).argmax(axis=1)# 获取预测的类别标签
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)# 计算准确率
        cur_log["epoch"] = wrapper.cur_epoch # 当前epoch
        cur_log["acc"] = acc # 当前epoch的准确率
        self.logs.append(cur_log) # 记录日志
        if self.verbose > 0:  # 打印当前epoch和准确率
            print("[Epoch {:5d}] Test ACC: {:.4f}".format(cur_log["epoch"], cur_log["acc"]))


class EarlyStoppingACC(Callback):
    """
    根据分类准确率进行早停。
    :param numpy.array X: 特征矩阵，大小为 :math:`[N, D]`，其中 :math:`N` 是样本数，:math:`D` 是特征数。
    :param numpy.array y: 标签矩阵，大小为 :math:`[N, 1]`。
    :param int patience: 如果在指定的epoch数量（patience）内没有提高，训练将停止。
    :param int verbose: 详细模式。
    :param str save_path: 如果 :code:`save_path=None`，则不保存模型，否则将最佳准确率的模型保存到指定路径。
    """
    def __init__(self, X, y, patience=1, verbose=0, save_path=None):
        super(EarlyStoppingACC, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.patience = patience
        self.best_acc = 0 # 最佳准确率初始化为0
        self.cnt = 0 # 最佳准确率初始化为0
        self.logs = [] # 计数器初始化为0
        self.save_path = save_path

    def on_epoch_end(self, wrapper): # 在每个epoch结束时计算准确率，并根据准确率决定是否进行早停
        cur_log = {}
        y_pred = wrapper.predict(self.X).argmax(axis=1)# 获取预测的类别标签
        acc = accuracy_score(y_true=self.y, y_pred=y_pred) # 计算准确率

        if acc > self.best_acc:# 如果当前准确率优于最佳准确率
            self.best_acc = acc# 更新最佳准确率
            self.cnt = 0
            if self.save_path is not None: # 如果指定了保存路径，则保存当前的最佳模型
                wrapper.save(self.save_path)
        else:# 如果当前准确率不优于最佳准确率
            self.cnt += 1# 计数器加1
            if self.cnt > self.patience:# 如果计数器超过了patience，触发早停
                wrapper.stop_training = True# 设置停止训练的标志
        cur_log["epoch"] = wrapper.cur_epoch# 当前epoch
        cur_log["acc"] = acc# 当前epoch的准确率
        cur_log["best_acc"] = self.best_acc# 当前最优的准确率
        self.logs.append(cur_log) # 记录日志
        if self.verbose > 0: # 打印当前epoch、准确率和最优准确率
            print("[Epoch {:5d}] EarlyStopping Callback ACC: {:.4f}, Best ACC: {:.4f}".format(cur_log["epoch"], cur_log["acc"], cur_log["best_acc"]))

