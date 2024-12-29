import torch
import numpy as np
from das_torch import das


def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """
    Lag-one coherence of the receive aperture.
    """
    iq = torch.transpose(iq, 0, 1)  # 将接收孔径数据移动到第 0 维
    # iq = iq.permute(1, 0, 2)
    # 计算对接收孔径的时间延迟修正
    # 对比das传入的参数：iqraw=iq, tA=t_rx, tB=t_tx, fs=fs, fd=fd, A=以接收孔径数量的单位阵
    rxdata = das(iq, t_rx, t_tx, fs, fd, torch.eye(iq.shape[0], device=iq.device))  # 时间对齐后的 IQ 数据

    # 计算相邻接收通道之间的相关系数
    xy = torch.real(torch.nansum(rxdata[:-1] * torch.conj(rxdata[1:]), dim=0))
    xx = torch.nansum(torch.abs(rxdata[:-1]) ** 2, dim=0)
    yy = torch.nansum(torch.abs(rxdata[1:]) ** 2, dim=0)
    ncc = xy / torch.sqrt(xx * yy)
    return ncc


def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """
    The coherence factor of the receive aperture.
    """
    iq = torch.transpose(iq, 0, 1)  # 将接收孔径数据移动到第 0 维
    # iq = iq.permute(1, 0, 2)
    rxdata = das(iq, t_rx, t_tx, fs, fd, torch.eye(iq.shape[0], device=iq.device))  # 时间对齐后的 IQ 数据

    # 计算相干因子
    num = torch.abs(torch.nansum(rxdata, dim=0))
    den = torch.nansum(torch.abs(rxdata), dim=0)
    return num / den


def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """
    The speckle brightness criterion.
    """
    return torch.nanmean(torch.abs(das(iq, t_tx, t_rx, fs, fd)))


def total_variation(c):
    """
    Total variation of the sound speed map `c` in x and z directions.
    """
    tvx = torch.nanmean(torch.square(torch.diff(c, dim=0)))
    # tvx = torch.nanmean(torch.square(c[:-1, :] - c[1:, :]))
    tvz = torch.nanmean(torch.square(torch.diff(c, dim=1)))
    # tvz = torch.nanmean(torch.square(c[:, :-1] - c[:, 1:]))
    return tvx + tvz


def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """
    Phase error between transmit and receive apertures.
    """
    # 提取 IQ 数据的维度信息
    nrx, ntx, nsamps = iq.shape

    # 创建子孔径掩码
    mask = torch.zeros((nrx, ntx), device=iq.device)
    halfsa = 8  # 子孔径半径
    dx = 1  # 子孔径步长
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + torch.diag(torch.ones((ntx - abs(diag),), device=iq.device), diag)
    mask = mask[halfsa : mask.shape[0] - halfsa : dx]
    At = torch.flip(mask, dims=(0,))  # 上下翻转
    Ar = mask

    # 子孔径聚焦数据
    iqfoc = das(iq, t_tx, t_rx, fs, fd, At, Ar)

    # 计算相位误差
    xy = iqfoc[:-1, :-1] * torch.conj(iqfoc[1:, 1:])
    xx = iqfoc[:-1, :-1] * torch.conj(iqfoc[:-1, :-1])
    yy = iqfoc[1:, 1:] * torch.conj(iqfoc[1:, 1:])

    # 过滤有效区域
    valid1 = (iqfoc[:-1, :-1] != 0) & (iqfoc[1:, 1:] != 0)
    xy = torch.tensor(np.where(valid1, np.where(valid1, xy, 0), 0))
    xx = torch.tensor(np.where(valid1, np.where(valid1, xx, 0), 0))
    yy = torch.tensor(np.where(valid1, np.where(valid1, yy, 0), 0))

    # 计算相关性平方并过滤阈值
    xy = torch.sum(xy, dim=-1)
    xx = torch.sum(xx, dim=-1)
    yy = torch.sum(yy, dim=-1)
    ccsq = torch.square(torch.abs(xy)) / (torch.abs(xx) * torch.abs(yy))
    valid2 = ccsq > thresh ** 2
    xy = torch.where(valid2, torch.where(valid2, xy, 0), 0)

    # 转换和重塑结果
    xy = torch.flip(xy, dims=(0,))  # 反对角线 -> 对角线
    xy = xy.reshape(*xy.shape[:2], -1)
    xy = xy.permute(2, 0, 1)  # 将子孔径维度移到内部
    xy = torch.triu(xy) + torch.transpose(torch.conj(torch.tril(xy)), 0, 2)
    dphi = torch.angle(xy)  # 计算相位
    return dphi
