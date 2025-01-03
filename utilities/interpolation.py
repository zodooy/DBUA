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

def interpolate(t, x0, z0, x1, z1, xc, zc, s):
    xt = t * (x1 - x0) + x0  # x在t间距的空间位置
    zt = t * (z1 - z0) + z0  # z在t间距的空间位置

    # 将空间位置转换为在慢度图中的xc和zc坐标中的索引
    dxc, dzc = xc[1] - xc[0], zc[1] - zc[0]  # 网格间距
    # 在慢度图中获取xt、zt的索引，并保证该索引始终在有效范围[0, s.shape[0] - 1]内
    xit = torch.clip((xt - xc[0]) / dxc, 0, s.shape[0] - 1)
    zit = torch.clip((zt - zc[0]) / dzc, 0, s.shape[1] - 1)
    # 找到左下邻居点和右上邻居点的索引
    xi0 = torch.floor(xit)
    zi0 = torch.floor(zit)
    xi1 = xi0 + 1
    zi1 = zi0 + 1

    # （xt，zt）处的插值慢速
    # 插值点周围的四个网格点分别为 (xi0, zi0)、(xi1, zi0)、(xi0, zi1) 和 (xi1, zi1)
    s00 = s[xi0.int(), zi0.int()]
    s10 = s[torch.clip(xi1, 0, s.shape[0] - 1).int(), zi0.int()]
    s01 = s[xi0.int(), torch.clip(zi1, 0, s.shape[0] - 1).int()]
    s11 = s[torch.clip(xi1, 0, s.shape[0] - 1).int(), torch.clip(zi1, 0, s.shape[0] - 1).int()]
    # 双线性插值
    w00 = (xi1 - xit) * (zi1 - zit)
    w10 = (xit - xi0) * (zi1 - zit)
    w01 = (xi1 - xit) * (zit - zi0)
    w11 = (xit - xi0) * (zit - zi0)
    return s00 * w00 + s10 * w10 + s01 * w01 + s11 * w11