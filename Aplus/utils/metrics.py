import numpy as np
import torch


def mae(y_hat: torch.Tensor, y: torch.Tensor):
    return torch.abs(y_hat - y).mean().detach().cpu()


def accuracy(y_hat: torch.Tensor, y: torch.Tensor):
    return float((y_hat - y).eq(0).float().mean().detach().cpu())


def matrix_norm(x: np.ndarray, dim):
    scale = np.sum(x, axis=dim, keepdims=True)
    return x / scale


def confusion_matrix(y_hat: torch.Tensor, y: torch.Tensor, n_classes, normalize=False):
    """
    Calculate confusion matrix.
    Args:
        y_hat: Pred label.
        y: True label
        n_classes: Number of classes.
        normalize: Return normalized confusion matrix.

    Returns:
        confusion matrix.

    """
    matrix = torch.zeros(size=(n_classes, n_classes))
    # 遍历所有类别
    for real_class in range(n_classes):
        # 真实类别为real_class的数量
        real_class_case_mask = (y-real_class).eq(0)
        n_real_class_case = real_class_case_mask.sum()
        if n_real_class_case == 0:
            continue
        for pred_class in range(n_classes):
            # 真实类别为real_class的数据被预测为pred_class的数量
            case_mask = real_class_case_mask * (y_hat-pred_class).eq(0)
            case_count = case_mask.sum()
            if normalize:
                matrix[real_class, pred_class] = case_count / n_real_class_case
            else:
                matrix[real_class, pred_class] = case_count

    return np.array(matrix)

