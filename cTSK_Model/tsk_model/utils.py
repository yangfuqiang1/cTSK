import torch

#check_tensor(tensor, dtype) 函数用于将输入的 numpy 数组或 torch 张量转换为指定数据类型的 torch 张量。
def check_tensor(tensor, dtype):
    """
    将 :code:`tensor` 转换为 :code:`dtype` 类型的 torch.Tensor.
    :param numpy.array/torch.tensor tensor: 输入数据。
    :param str dtype: PyTorch 数据类型字符串。
    :return: 一个 :code:`dtype` 类型的 torch.Tensor。
    """
    return torch.tensor(tensor, dtype=dtype)

#reset_params(model) 函数用于重置给定模型的所有参数。
# 如果模型本身有 reset_parameters 方法，则直接调用该方法；否则，递归地对模型的子层进行参数重置。
def reset_params(model):
    """
    重置 :code:`model` 中的所有参数。
    :param torch.nn.Module model: Pytorch 模型。
    """
    if hasattr(model, "reset_parameters"):
        model.reset_parameters()
    else:
        for layer in model.children():
            reset_params(layer)

#NumpyDataLoader 类用于将 numpy 数组转换为可以被 PyTorch DataLoader 使用的数据加载器。
# 它实现了 __len__ 和 __getitem__ 方法，使得该类可以像数据加载器一样在训练过程中被迭代。
class NumpyDataLoader:
    """
    将 numpy 数组转换为数据加载器。
    :param numpy.array *inputs: numpy 数组。
    """
    def __init__(self, *inputs):
        self.inputs = inputs
        self.n_inputs = len(inputs)

    def __len__(self):
        return self.inputs[0].shape[0]

    def __getitem__(self, item):
        if self.n_inputs == 1:
            return self.inputs[0][item]
        else:
            return [array[item] for array in self.inputs]