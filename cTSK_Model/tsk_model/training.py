import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.special import softmax
from torch.utils.data import DataLoader
from .utils import NumpyDataLoader


def ur_loss(frs, tau=0.5):
    """
    由Cui等人提出的均匀正则化（UR）[3]。
    UR损失计算公式为 :math:`\ell_{UR} = \sum_{r=1}^R (\frac{1}{N}\sum_{n=1}^N f_{n,r} - \tau)^2`，
    其中 :math:`f_{n,r}` 表示第 :math:`n` 个样本在第 :math:`r` 条规则上的触发水平。

    :param torch.tensor frs: 触发水平矩阵（来自前件的输出），大小为 :math:`[N, R]`，
        其中 :math:`N` 表示样本数量，:math:`R` 表示规则数量。
    :param float tau: 每条规则平均触发水平的期望值 :math:`\tau`。对于一个 :math:`C` 类的分类问题，
        我们建议将 :math:`\tau` 设置为 :math:`1/C`，对于回归问题，:math:`\tau` 可以设置为 :math:`0.5`。
    :return: 一个标量值，代表UR损失。
    """
    return ((torch.mean(frs, dim=0) - tau) ** 2).sum()


def causal_regularization_loss(model, causal_weights, X, alpha=0.1):
    """
    因果正则化损失：鼓励模型使用因果重要的特征

    :param torch.nn.Module model: TSK模型
    :param numpy.array causal_weights: 先验因果权重，大小为 [D]
    :param torch.tensor X: 输入数据，大小为 [N, D]
    :param float alpha: 正则化强度
    :return: 因果正则化损失
    """
    device = X.device

    # 保存原始模式并设置为训练模式以确保梯度计算
    original_mode = model.training
    model.train()

    try:
        # 创建需要梯度的输入副本
        X_with_grad = X.clone().detach().requires_grad_(True)

        # 前向传播
        output = model(X_with_grad)

        # 检查输出形状并创建正确的梯度目标
        if output.dim() == 1:
            # 单输出
            grad_output = torch.ones_like(output, device=device)
        elif output.dim() == 2:
            # 多输出（多分类），对每个样本的所有输出求和
            grad_output = torch.ones(output.shape[0], output.shape[1], device=device)
        else:
            # 其他情况，使用单位矩阵
            grad_output = torch.eye(output.shape[1], device=device).repeat(output.shape[0], 1, 1)

        # 计算输出对输入的梯度
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=X_with_grad,
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
            allow_unused=True
        )[0]

        if gradients is None:
            return torch.tensor(0.0, device=device)

        # 处理多输出情况：对每个特征的所有输出梯度取平均
        if gradients.dim() == 3:
            # [batch_size, num_classes, num_features] -> [batch_size, num_features]
            gradients = gradients.mean(dim=1)
        elif gradients.dim() == 2 and gradients.shape[1] != X.shape[1]:
            # 如果梯度形状不匹配，可能是多输出情况
            gradients = gradients.mean(dim=1, keepdim=True).expand(-1, X.shape[1])

        # 平均绝对梯度作为特征重要性
        feature_importance = gradients.abs().mean(dim=0)

        # 归一化特征重要性
        feature_importance_norm = feature_importance / (feature_importance.sum() + 1e-8)

        # 将先验因果权重转换为tensor并移动到正确设备
        causal_weights_tensor = torch.tensor(causal_weights, dtype=torch.float32, device=device)

        # 归一化因果权重
        causal_weights_norm = causal_weights_tensor / (causal_weights_tensor.sum() + 1e-8)

        # 计算一致性损失（MSE）
        loss = torch.mean((feature_importance_norm - causal_weights_norm) ** 2)

        return alpha * loss

    except Exception as e:
        print(f"因果正则化计算警告: {e}")
        return torch.tensor(0.0, device=device)
    finally:
        # 恢复模型原始模式
        model.train(original_mode)


def causal_regularization_loss_v2(model, causal_weights, X, alpha=0.1):
    """
    改进版本的因果正则化损失：分别计算每个输出类别的梯度
    """
    device = X.device
    original_mode = model.training
    model.train()

    try:
        X_with_grad = X.clone().detach().requires_grad_(True)
        output = model(X_with_grad)

        batch_size, num_classes = output.shape

        # 为每个类别计算梯度
        all_gradients = []
        for class_idx in range(num_classes):
            # 创建针对特定类别的梯度目标
            grad_output = torch.zeros_like(output, device=device)
            grad_output[:, class_idx] = 1.0

            gradients = torch.autograd.grad(
                outputs=output,
                inputs=X_with_grad,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=True,  # 保持计算图以便多次计算
                only_inputs=True,
                allow_unused=True
            )[0]

            if gradients is not None:
                all_gradients.append(gradients.abs().mean(dim=0))

        if not all_gradients:
            return torch.tensor(0.0, device=device)

        # 平均所有类别的梯度重要性
        feature_importance = torch.stack(all_gradients).mean(dim=0)
        feature_importance_norm = feature_importance / (feature_importance.sum() + 1e-8)

        causal_weights_tensor = torch.tensor(causal_weights, dtype=torch.float32, device=device)
        causal_weights_norm = causal_weights_tensor / (causal_weights_tensor.sum() + 1e-8)

        loss = torch.mean((feature_importance_norm - causal_weights_norm) ** 2)
        return alpha * loss

    except Exception as e:
        print(f"因果正则化计算警告: {e}")
        return torch.tensor(0.0, device=device)
    finally:
        model.train(original_mode)


def causal_regularization_loss_v3(model, causal_weights, X, alpha=0.1):
    """
    简化版本的因果正则化损失：使用输出总和计算梯度
    """
    device = X.device
    original_mode = model.training
    model.train()

    try:
        X_with_grad = X.clone().detach().requires_grad_(True)
        output = model(X_with_grad)

        # 对每个样本的所有输出求和，然后计算梯度
        output_sum = output.sum(dim=1) if output.dim() > 1 else output
        grad_output = torch.ones_like(output_sum, device=device)

        gradients = torch.autograd.grad(
            outputs=output_sum,
            inputs=X_with_grad,
            grad_outputs=grad_output,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
            allow_unused=True
        )[0]

        if gradients is None:
            return torch.tensor(0.0, device=device)

        feature_importance = gradients.abs().mean(dim=0)
        feature_importance_norm = feature_importance / (feature_importance.sum() + 1e-8)

        causal_weights_tensor = torch.tensor(causal_weights, dtype=torch.float32, device=device)
        causal_weights_norm = causal_weights_tensor / (causal_weights_tensor.sum() + 1e-8)

        loss = torch.mean((feature_importance_norm - causal_weights_norm) ** 2)
        return alpha * loss

    except Exception as e:
        print(f"因果正则化计算警告: {e}")
        return torch.tensor(0.0, device=device)
    finally:
        model.train(original_mode)


class CausalWrapper:
    """
    因果加权的TSK训练包装器
    """

    def __init__(self, model, optimizer, criterion, causal_weights,
                 causal_reg_weight=0.1, batch_size=512, epochs=1,
                 callbacks=None, label_type="c", device="cuda",
                 reset_param=True, ur=0, ur_tau=0.5, causal_version=3, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.causal_weights = causal_weights
        self.causal_reg_weight = causal_reg_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.label_type = label_type
        self.ur = ur
        self.ur_tau = ur_tau
        self.causal_version = causal_version  # 选择因果正则化版本

        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            raise ValueError("callback must be a Callback object")

        self.reset_param = reset_param
        if self.reset_param:
            self.model.reset_parameters()

        self.cur_batch = 0
        self.cur_epoch = 0
        self.kwargs = kwargs
        self.stop_training = False

        print(f"使用设备: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU名称: {torch.cuda.get_device_name(self.device)}")

    def train_on_batch(self, input, target):
        """
        使用因果正则化的批次训练
        """
        input, target = input.to(self.device), target.to(self.device)

        # 前向传播
        outputs, frs = self.model(input, get_frs=True)

        # 基础损失
        base_loss = self.criterion(outputs, target)

        # UR损失
        ur_loss_value = ur_loss(frs, self.ur_tau) if self.ur > 0 else torch.tensor(0.0, device=self.device)

        # 因果正则化损失
        causal_reg_loss = torch.tensor(0.0, device=self.device)
        if self.causal_reg_weight > 0 and self.model.training:
            try:
                if self.causal_version == 1:
                    causal_reg_loss = causal_regularization_loss(
                        self.model, self.causal_weights, input, self.causal_reg_weight
                    )
                elif self.causal_version == 2:
                    causal_reg_loss = causal_regularization_loss_v2(
                        self.model, self.causal_weights, input, self.causal_reg_weight
                    )
                else:  # version 3
                    causal_reg_loss = causal_regularization_loss_v3(
                        self.model, self.causal_weights, input, self.causal_reg_weight
                    )
            except Exception as e:
                print(f"因果正则化计算失败: {e}")
                causal_reg_loss = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = base_loss + self.ur * ur_loss_value + causal_reg_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    # 其他方法保持不变（fit, fit_loader, predict, predict_proba, 等）
    # ... 保持原有的 fit, fit_loader, predict 等方法不变 ...

    def fit(self, X, y):
        """
        使用numpy数组训练模型
        """
        X = X.astype("float32")
        if self.label_type == "c":
            y = y.astype("int64")
        elif self.label_type == "r":
            y = y.astype("float32")
        else:
            raise ValueError("label_type can only be \"c\" or \"r\"!")

        train_loader = DataLoader(
            NumpyDataLoader(X, y),
            batch_size=self.batch_size,
            shuffle=self.kwargs.get("shuffle", True),
            num_workers=self.kwargs.get("num_workers", 0),
            drop_last=self.kwargs.get("drop_last", True if self.batch_size < X.shape[0] else False),
            pin_memory=True
        )

        self.fit_loader(train_loader)
        return self

    def fit_loader(self, train_loader):
        """
        使用数据加载器训练模型
        """
        self.stop_training = False
        for e in range(self.epochs):
            self.cur_epoch = e
            self.__run_callbacks__("on_epoch_begin")

            epoch_losses = []
            for batch_idx, inputs in enumerate(train_loader):
                self.__run_callbacks__("on_batch_begin")

                self.model.train()
                loss = self.train_on_batch(inputs[0], inputs[1])
                epoch_losses.append(loss)

                if batch_idx % 100 == 0:
                    print(f"Epoch {e + 1}/{self.epochs}, Batch {batch_idx}, Loss: {loss:.4f}")

                self.__run_callbacks__("on_batch_end")
                self.cur_batch += 1

            avg_loss = np.mean(epoch_losses)
            print(f"Epoch {e + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")

            self.__run_callbacks__("on_epoch_end")
            if self.stop_training:
                print("训练提前停止")
                break

        return self

    def predict(self, X, y=None):
        """
        预测
        """
        X = X.astype("float32")
        test_loader = DataLoader(
            NumpyDataLoader(X),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.kwargs.get("num_workers", 0),
            drop_last=False,
            pin_memory=True
        )

        y_preds = []
        self.model.eval()
        with torch.no_grad():
            for inputs in test_loader:
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                inputs = inputs.to(self.device)
                y_pred = self.model(inputs).detach().cpu().numpy()
                y_preds.append(y_pred)

        return np.concatenate(y_preds, axis=0)

    # 其他方法保持不变...

    def predict_proba(self, X, y=None):
        """
        预测概率（仅分类）
        """
        if self.label_type == "r":
            raise ValueError("predict_proba can only be used when label_type=\"c\"")

        y_preds = self.predict(X)
        return softmax(y_preds, axis=1)

    def __run_callbacks__(self, func_name):
        for cb in self.callbacks:
            getattr(cb, func_name)(self)

    def save(self, path):
        """
        保存模型
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'causal_weights': self.causal_weights,
            'causal_reg_weight': self.causal_reg_weight
        }, path)

    def load(self, path):
        """
        加载模型
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.causal_weights = checkpoint['causal_weights']
        self.causal_reg_weight = checkpoint['causal_reg_weight']

    def get_causal_weights(self):
        """
        获取当前的因果权重
        """
        if hasattr(self.model.antecedent, 'get_causal_weights'):
            return self.model.antecedent.get_causal_weights()
        return None

    def evaluate_feature_importance(self, X):
        """
        评估当前模型的特征重要性
        """
        X_tensor = torch.tensor(X.astype("float32"), device=self.device)

        self.model.eval()
        with torch.no_grad():
            X_with_grad = X_tensor.clone().detach().requires_grad_(True)
            output = self.model(X_with_grad)

            if output.dim() == 1:
                grad_output = torch.ones_like(output, device=self.device)
            else:
                grad_output = torch.ones(output.shape[0], device=self.device)
                if output.dim() > 1:
                    grad_output = grad_output.unsqueeze(1)

            gradients = torch.autograd.grad(
                outputs=output,
                inputs=X_with_grad,
                grad_outputs=grad_output,
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]

            if gradients is not None:
                feature_importance = gradients.abs().mean(dim=0).cpu().numpy()
                return feature_importance / (feature_importance.sum() + 1e-8)

        return None