import torch
from torch.func import vmap

def das(iqraw, tA, tB, fs, fd, A=None, B=None, apoA=1, apoB=1, interp="cubic"):
    """
    Delay-and-sum IQ data according to a given time delay profile.

    @param iqraw   [na, nb, nsamps]  Raw IQ data (baseband)
    @param tA      [na, *pixdims]    Time delays to apply to dimension 0 of iq
    @param tB      [nb, *pixdims]    Time delays to apply to dimension 1 of iq
    @param fs      scalar            Sampling frequency to convert from time to samples
    @param fd      scalar            Demodulation frequency (0 for RF modulated data)
    @param A       [*na_out, na]     Linear combination of dimension 0 of iqraw
    @param B       [*nb_out, nb]     Linear combination of dimension 1 of iqraw
    @param apoA    [na, *pixdims]    Broadcastable apodization on dimension 0 of iq
    @param apoB    [nb, *pixdims]    Broadcastable apodization on dimension 1 of iq
    @param interp  string            Interpolation method to use
    @return iqfoc  [*na_out, *nb_out, *pixel_dims]   Beamformed IQ data
    """
    # 默认的线性组合是对所有元素求和
    if A is None:
        A = torch.ones(iqraw.shape[0], device=iqraw.device)
    if B is None:
        B = torch.ones(iqraw.shape[1], device=iqraw.device)

    # 选择插值函数
    fints = {
        "nearest": interp_nearest,
        "linear": interp_linear,
        "cubic": interp_cubic,
        "lanczos3": lambda x, t: interp_lanczos(x, t, nlobe=3),
        "lanczos5": lambda x, t: interp_lanczos(x, t, nlobe=5),
    }
    fint = fints[interp]

    # 基带插值，将iq数据按照时间延迟t进行插值
    def bbint(iq, t):
        iqfoc = fint(iq, fs * t) # 插值时间通过fs * t转换为采样点位置
        # print(iqfoc)
        return iqfoc * torch.exp(2j * torch.pi * fd * t) # 对插值后的 IQ 数据进行基带调制

    # 内层函数：处理单个接收通道
    def das_b(x):
        iq_i, tA_i = x
        return torch.tensordot(B.to(torch.complex64), vmap(bbint)(iq_i, tA_i + tB) * apoB, dims=([-1], [0]))

    # 外层函数：处理多个发射通道
    return torch.tensordot(A.to(torch.complex64), torch.stack([das_b(x) for x in zip(iqraw, tA)]) * apoA, dims=([-1], [0]))


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
