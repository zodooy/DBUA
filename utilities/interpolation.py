import torch

# 安全访问，避免访问越界
def safe_access(x, s):
    """Safe access to array x at indices s."""
    s = s.long()
    valid = (s >= 0) & (s < x.size(0))
    return x[s * valid] * valid

# 最近邻插值
def interp_nearest(x, si):
    """1D nearest neighbor interpolation with torch."""
    return x[torch.clip(torch.round(si).long(), 0, x.size(0) - 1)]

# 线性插值
def interp_linear(x, si):
    """1D linear interpolation with torch."""
    s = torch.floor(si)  # 提取整数部分
    f = si - s  # 提取小数部分
    x0 = safe_access(x, s + 0)
    x1 = safe_access(x, s + 1)
    return (1 - f) * x0 + f * x1

# 三次 Hermite 插值
def interp_cubic(x, si):
    # print(si)
    """1D cubic Hermite interpolation with torch."""
    s = torch.floor(si)  # 提取整数部分
    f = si - s  # 提取小数部分
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)

    # 插值权重
    a0 = f * (-1 + f * (+2 * f - 1))
    a1 = 2 + f * (+0 + f * (-5 * f + 3))
    a2 = f * (+1 + f * (+4 * f - 3))
    a3 = f * (+0 + f * (-1 * f + 1))
    return (a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3) / 2

# Lanczos 核
def _lanczos_helper(x, nlobe=3):
    """Lanczos kernel."""
    a = (nlobe + 1) / 2
    return torch.where(torch.abs(x) < a, torch.sinc(x) * torch.sinc(x / a), torch.tensor(0.0, device=x.device))

def interp_lanczos(x, si, nlobe=3):
    """Lanczos interpolation with torch."""
    s = torch.floor(si)  # 提取整数部分
    f = si - s  # 提取小数部分
    x0 = safe_access(x, s - 1)
    x1 = safe_access(x, s + 0)
    x2 = safe_access(x, s + 1)
    x3 = safe_access(x, s + 2)

    a0 = _lanczos_helper(f + 1, nlobe)
    a1 = _lanczos_helper(f + 0, nlobe)
    a2 = _lanczos_helper(f - 1, nlobe)
    a3 = _lanczos_helper(f - 2, nlobe)
    return a0 * x0 + a1 * x1 + a2 * x2 + a3 * x3