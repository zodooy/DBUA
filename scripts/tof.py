import torch
from utilities.interpolation import interpolate

def time_of_flight(x0, z0, x1, z1, xc, zc, c, fnum: float, npts: int, Dmin: float):
    """
    根据在网格点（xc，zc）上定义的声速图c，获得从（x0，z0）到（x1，z1）的ToF。
    x0:     [...]       Path origin in x (arbitrary dimensions, broadcasting allowed)
    z0:     [...]       Path origin in z (arbitrary dimensions, broadcasting allowed)
    x1:     [...]       Path finish in x (arbitrary dimensions, broadcasting allowed)
    z1:     [...]       Path finish in z (arbitrary dimensions, broadcasting allowed)
    xc:     [nxc,]      Vector of x-grid points in sound speed definition (c.shape[0],)
    zc:     [nzc,]      Vector of x-grid points in sound speed definition
    c:      [nxc, nzc]  Sound speed map in (xc, zc) coordinates
    fnum:   scalar      f-number to apply
    npts:   scalar      Number of points in time-of-flight line segment
    Dmin:   scalar      Minimum size of the aperture, regardless of f-number
    """

    # 创建路径采样点 t_all
    t_all = torch.linspace(1, 0, npts + 1, device=x0.device)[:-1].flip(0)

    # 定义慢速图 (slowness map)
    s = 1 / c

    # 计算飞行时间 (ToF)
    dx = torch.abs(x1 - x0)
    dz = torch.abs(z1 - z0)
    dtrue = torch.sqrt(dx**2 + dz**2)

    # 对所有采样点 t_all 应用插值函数
    slowness = torch.stack([interpolate(t, x0, z0, x1, z1, xc, zc, s) for t in t_all])
    # tof 作为 slowness 的均值乘以距离 dtrue
    tof = torch.nanmean(slowness, dim=0) * dtrue

    # F-number mask for valid points（有效点条件1：焦点的横向宽度不应过大）
    fnum_valid = torch.abs(2 * fnum * dx) <= dz
    # 将最小孔径宽度设置为3mm，Dmin=3e-3（有效点条件2：垂直距离和水平距离都需要小于最小孔径限制）
    Dmin_valid = torch.logical_and(dz < Dmin * fnum, dx < Dmin / 2)
    # Total mask for valid regions（有效条件1与有效条件2合并）
    valid = torch.logical_or(fnum_valid, Dmin_valid)

    # 对无效区域指定默认值
    # 对于无效区域，指定虚拟TOF，标记后之后会插值为0
    tof_valid = torch.where(valid, tof, torch.tensor(1.0, device=tof.device))
    # 为无效点设置默认值为-10（标记为无效的传播时间）
    tof = torch.where(valid, tof_valid, torch.tensor(-10.0, device=tof.device))

    return tof
