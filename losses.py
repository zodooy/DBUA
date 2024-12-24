from functools import partial

import jax.numpy as jnp
import numpy as np


from das import das


def lag_one_coherence(iq, t_tx, t_rx, fs, fd):
    """
    Lag-one coherence of the receive aperture (DOI: 10.1109/TUFFC.2018.2855653).
    LOC测量信号相对于其噪声的质量，可用于选择声输出
    """
    iq = jnp.transpose(iq, (1, 0, 2))  # 将接收孔径放在第0维
    # 计算对接收孔径的时间延迟修正
    # 对比das传入的参数：iqraw=iq, tA=t_rx, tB=t_tx, fs=fs, fd=fd, A=以接收孔径数量的单位阵
    rxdata = das(iq, t_rx, t_tx, fs, fd, jnp.eye(iq.shape[0])) # 经过时间对齐后，各接收通道的IQ数据
    # 计算相关系数
    xy = jnp.real(jnp.nansum(rxdata[:-1] * jnp.conj(rxdata[1:]), axis=0))
    xx = jnp.nansum(jnp.abs(rxdata[:-1]) ** 2, axis=0)
    yy = jnp.nansum(jnp.abs(rxdata[1:]) ** 2, axis=0)
    ncc = xy / jnp.sqrt(xx * yy)
    return ncc


def coherence_factor(iq, t_tx, t_rx, fs, fd):
    """
    The coherence factor of the receive aperture (DOI: 10.1121/1.410562).
    The CF is a focusing criterion used to measure the amount of aberration in an image.
    """
    iq = jnp.transpose(iq, (1, 0, 2))  # 接收孔径数据移动到第0维
    rxdata = das(iq, t_rx, t_tx, fs, fd, jnp.eye(iq.shape[0]))  # 时间对齐后的接收通道数据
    # 计算相干因子
    num = jnp.abs(jnp.nansum(rxdata, axis=0))
    den = jnp.nansum(jnp.abs(rxdata), axis=0)
    return num / den


def speckle_brightness(iq, t_tx, t_rx, fs, fd):
    """
    The speckle brightness criterion (DOI: 10.1121/1.397889)
    散斑亮度可用于测量聚焦质量。
    """
    return jnp.nanmean(jnp.abs(das(iq, t_tx, t_rx, fs, fd)))


def total_variation(c):
    """
    x和z中声速图的总变化。
    声速图c应指定为大小为[nx，nz]的2D矩阵
    ·计算矩阵c在x方向的相邻元素差值
    ·计算相邻差值的平方
    ·对所有差值的平方进行平均，忽略 NaN 值
    """
    tvx = jnp.nanmean(jnp.square(jnp.diff(c, axis=0)))
    tvz = jnp.nanmean(jnp.square(jnp.diff(c, axis=1)))
    return tvx + tvz


def phase_error(iq, t_tx, t_rx, fs, fd, thresh=0.9):
    """
    发射和接收孔径之间的相位误差
    This error is closesly related to the "Translated Transmit Apertures" algorithm
    (DOI: 10.1109/58.585209), where translated transmit and receive apertures
    with common midpoint should have perfect speckle correlation by the van
    Cittert Zernike theorem (DOI: 10.1121/1.418235). High correlation will
    result in high-quality phase shift estimates (DOI: 10.1121/10.0000809).
    CUTE also takes a similar approach (DOI: 10.1016/j.ultras.2020.106168),
    but in the angular basis instead of the element basis.
    """
    # 计算给定发射和接收子孔径的IQ数据。
    # IQ数据矩阵如下：（发射孔径索引，接收孔径索引）
    #   A B C    A: (2, 0)   B: (2, 1)   C: (2, 2)
    #   D E F    D: (1, 0)   E: (1, 1)   F: (1, 2)
    #   G H I    G: (0, 0)   H: (0, 1)   I: (0, 2)
    # 对角线对应具有共同中点的tx/rx对, 例如:
    #   A, E, and I have a midpoint at 1.
    #   D and H have a midpoint at 0.5.
    #   G has a midpoint at 0.
    #   B and F have a midpoint at 1.5.
    #   C has a midpoint at 2.
    #
    # We create tx and rx subapertures of size 2*halfsa+1 elements, with
    # spacing determined by dx. These are made using das_subap.

    # 提取iq三个维度的信息，分别为：接收通道数、发射通道数、时间采样点数
    nrx, ntx, nsamps = iq.shape

    # 子孔径掩码生成
    # 二维掩码初始化为0
    mask = np.zeros((nrx, ntx))
    halfsa = 8  # 子孔径的半径
    dx = 1  # 子孔径的步长
    for diag in range(-halfsa, halfsa + 1):
        mask = mask + jnp.diag(jnp.ones((ntx - abs(diag),)), diag)
    mask = mask[halfsa : mask.shape[0] - halfsa : dx]
    At = mask[::-1] # mask的上下翻转
    Ar = mask
    # 通过At和Ar的应用，对子孔径内的信号进行加权和求和，生成聚焦后的数据
    iqfoc = das(iq, t_tx, t_rx, fs, fd, At, Ar)

    # If <A,B> is the correlation between A and B, we want
    # <A, E>, <E, I>, <B, F>, <D, H>. The corners are naturally cut off.
    # 计算具有共同中点的相邻脉冲回波信号之间的相关性
    xy = iqfoc[:-1, :-1] * jnp.conj(iqfoc[+1:, +1:])
    xx = iqfoc[:-1, :-1] * jnp.conj(iqfoc[:-1, :-1])
    yy = iqfoc[+1:, +1:] * jnp.conj(iqfoc[+1:, +1:])
    # Use jax "double where" trick to remove correlations with only one signal
    valid1 = (iqfoc[:-1, :-1] != 0) & (iqfoc[1:, 1:] != 0)
    xy = jnp.where(valid1, jnp.where(valid1, xy, 0), 0)
    xx = jnp.where(valid1, jnp.where(valid1, xx, 0), 0)
    yy = jnp.where(valid1, jnp.where(valid1, yy, 0), 0)
    # Determine where the correlation coefficient is high enough to use
    xy = jnp.sum(xy, axis=-1)  # Sum over kernel
    xx = jnp.sum(xx, axis=-1)  # Sum over kernel
    yy = jnp.sum(yy, axis=-1)  # Sum over kernel
    ccsq = jnp.square(jnp.abs(xy)) / (jnp.abs(xx) * jnp.abs(yy))
    valid2 = ccsq > thresh * thresh
    xy = jnp.where(valid2, jnp.where(valid2, xy, 0), 0)
    # Convert
    xy = xy[::-1]  # 反对角线-->对角线
    xy = jnp.reshape(xy, (*xy.shape[:2], -1))
    xy = jnp.transpose(xy, (2, 0, 1))  # Place subap dimensions inside
    xy = jnp.triu(xy) + jnp.transpose(jnp.conj(jnp.tril(xy)), (0, 2, 1))
    dphi = jnp.angle(xy)  # 计算相移
    return dphi
