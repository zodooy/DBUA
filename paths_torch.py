import torch


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
