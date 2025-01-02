from utilities.data import *
import numpy as np
from utilities.plot import (
    plot_errors_vs_sound_speeds,
    imagesc
)
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch
from das import das
from tof import time_of_flight
from utilities.losses import (
    lag_one_coherence,
    coherence_factor,
    phase_error,
    total_variation,
    speckle_brightness,
)
import torch.optim as optim


def to_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def dbua(sample, loss_name, device):
    # Get IQ data, time zeros, sampling and demodulation frequency, and element positions
    iqdata, t0, fs, fd, elpos, _, _ = load_dataset(sample)
    xe, _, ze = np.array(elpos)
    iqdata = torch.tensor(iqdata, device=device)
    xe = torch.tensor(xe, device=device)
    ze = torch.tensor(ze, device=device)
    elpos = torch.tensor(elpos, device=device)
    t0 = torch.tensor(t0, device=device)
    wl0 = ASSUMED_C / fd  # wavelength (λ)

    # B-mode image dimensions
    xi = torch.arange(BMODE_X_MIN, BMODE_X_MAX, wl0 / 3)
    zi = torch.arange(BMODE_Z_MIN, BMODE_Z_MAX, wl0 / 3)
    nxi, nzi = xi.size(0), zi.size(0)
    xi, zi = torch.meshgrid(xi, zi, indexing="ij")
    xi, zi = to_cuda(xi), to_cuda(zi)

    # Sound speed grid dimensions
    xc = torch.linspace(SOUND_SPEED_X_MIN, SOUND_SPEED_X_MAX, SOUND_SPEED_NXC)
    zc = torch.linspace(SOUND_SPEED_Z_MIN, SOUND_SPEED_Z_MAX, SOUND_SPEED_NZC)
    dxc, dzc = to_cuda(xc[1] - xc[0]), to_cuda(zc[1] - zc[0])
    xc, zc = to_cuda(xc), to_cuda(zc)

    # Kernels to use for loss calculations (2λ x 2λ patches)
    xk, zk = torch.meshgrid((torch.arange(NXK) - (NXK - 1) / 2) * wl0 / 2,
                            (torch.arange(NZK) - (NZK - 1) / 2) * wl0 / 2, indexing="ij")
    xk, zk = to_cuda(xk), to_cuda(zk)

    # Kernel patch centers, distributed throughout the field of view
    xpc, zpc = torch.meshgrid(torch.linspace(PHASE_ERROR_X_MIN, PHASE_ERROR_X_MAX, NXP),
                              torch.linspace(PHASE_ERROR_Z_MIN, PHASE_ERROR_Z_MAX, NZP), indexing="ij")
    xpc, zpc = to_cuda(xpc), to_cuda(zpc)

    # Explicit broadcasting. Dimensions will be [elements, pixels, patches]
    xe = torch.reshape(xe, (-1, 1, 1))
    ze = torch.reshape(ze, (-1, 1, 1))
    xp = torch.reshape(xpc, (1, -1, 1)) + torch.reshape(xk, (1, 1, -1))
    zp = torch.reshape(zpc, (1, -1, 1)) + torch.reshape(zk, (1, 1, -1))
    xp = xp + 0 * zp  # Manual broadcasting
    zp = zp + 0 * xp  # Manual broadcasting
    xe, ze = to_cuda(xe), to_cuda(ze)
    xp, zp = to_cuda(xp), to_cuda(zp)


    # Compute time-of-flight for each {image, patch} pixel to each element
    def tof_image(c):
        return time_of_flight(xe, ze, xi, zi, xc, zc, c, fnum=0.5, npts=64, Dmin=3e-3)

    def tof_patch(c):
        return time_of_flight(xe, ze, xp, zp, xc, zc, c, fnum=0.5, npts=64, Dmin=3e-3)

    def makeImage(c):
        t = tof_image(c)
        return torch.abs(das(iqdata, t - t0, t, fs, fd))


    # Define loss functions
    def loss_wrapper(func, c):
        t = tof_patch(c)
        return func(iqdata, t - t0, t, fs, fd)

    sb_loss = lambda c: 1 - loss_wrapper(speckle_brightness, c)
    lc_loss = lambda c: 1 - torch.mean(loss_wrapper(lag_one_coherence, c))
    cf_loss = lambda c: 1 - torch.mean(loss_wrapper(coherence_factor, c))
    def pe_loss(c):
        t = tof_patch(c)
        dphi = phase_error(iqdata, t - t0, t, fs, fd)
        valid = dphi != 0
        dphi = torch.where(valid, dphi, torch.nan)
        return torch.nanmean(torch.log1p(torch.square(100 * dphi)))

    tv = lambda c: total_variation(c) * dxc * dzc

    def loss(c):
        if loss_name == "sb":  # Speckle brightness
            return sb_loss(c) + tv(c) * 1e2
        elif loss_name == "lc":  # Lag one coherence
            return lc_loss(c) + tv(c) * 1e2
        elif loss_name == "cf":  # Coherence factor
            return cf_loss(c) + tv(c) * 1e2
        elif loss_name == "pe":  # Phase error
            return pe_loss(c) + tv(c) * 1e2
        else:
            assert False


    # Initial survey of losses vs. global sound speed
    c = to_cuda(ASSUMED_C * torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)))

    # find optimal global sound speed for initialization
    print("Finding optimal global sound speed for initialization...")
    c0 = to_cuda(torch.linspace(1340, 1740, 201))
    dsb = torch.tensor([sb_loss(cc * to_cuda(torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)))) for cc in c0])
    dlc = torch.tensor([lc_loss(cc * to_cuda(torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)))) for cc in c0])
    dcf = torch.tensor([cf_loss(cc * to_cuda(torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)))) for cc in c0])
    dpe = torch.tensor([pe_loss(cc * to_cuda(torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)))) for cc in c0])
    # Use the sound speed with the optimal phase error to initialize sound speed map
    c = c0[torch.argmin(dpe)] * to_cuda(torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC)))

    # Plot global sound speed error
    plot_errors_vs_sound_speeds(c0.cpu(), dsb.cpu(), dlc.cpu(), dcf.cpu(), dpe.cpu(), sample)


    # Create the optimizer
    # c = 1490.0 * to_cuda(torch.ones((SOUND_SPEED_NXC, SOUND_SPEED_NZC))) # test sound speed
    c = c.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([c], lr=LEARNING_RATE, amsgrad=True)

    # Create the figure writer
    fig, _ = plt.subplots(1, 2, figsize=[9, 4])
    vobj = FFMpegWriter(fps=30)
    vobj.setup(fig, "videos/%s_opt%s.mp4" % (sample, loss_name), dpi=144)

    # Create the image axes for plotting
    ximm = xi[:, 0] * 1e3
    zimm = zi[0, :] * 1e3
    xcmm = xc * 1e3
    zcmm = zc * 1e3
    bdr = [-45, +5]
    cdr = np.array([-50, +50]) + \
        CTRUE[sample] if CTRUE[sample] > 0 else [1400, 1600]
    cmap = "seismic" if CTRUE[sample] > 0 else "jet"

    # Create a nice figure on first call, update on subsequent calls
    @torch.no_grad()
    def makeFigure(cimg, i, handles=None):
        b = makeImage(cimg)
        if handles is None:
            bmax = torch.max(b)
        else:
            hbi, hci, hbt, hct, bmax = handles
        bimg = b / bmax
        bimg = bimg + 1e-10 * (bimg == 0)  # Avoid nans
        bimg = 20 * torch.log10(bimg)
        bimg = torch.reshape(bimg, (nxi, nzi)).T
        cimg = torch.reshape(cimg, (SOUND_SPEED_NXC, SOUND_SPEED_NZC)).T

        if handles is None:
            # On the first time, create the figure
            fig.clf()
            plt.subplot(121)
            hbi = imagesc(ximm.cpu(), zimm.cpu(), bimg.cpu(), bdr, cmap="bone", interpolation="bicubic")
            hbt = plt.title("SB: %.2f, CF: %.3f, PE: %.3f" % (sb_loss(c), cf_loss(c), pe_loss(c)))
            plt.xlim(ximm[0].cpu(), ximm[-1].cpu())
            plt.ylim(zimm[-1].cpu(), zimm[0].cpu())
            plt.subplot(122)
            hci = imagesc(xcmm.cpu(), zcmm.cpu(), cimg.cpu(), cdr, cmap=cmap, interpolation="bicubic")
            if CTRUE[sample] > 0:  # When ground truth is provided, show the error
                hct = plt.title("Iteration %d: MAE %.2f" % (i, np.mean(np.abs(cimg.cpu() - CTRUE[sample]))))
            else:
                hct = plt.title("Iteration %d: Mean value %.2f" % (i, torch.mean(cimg)))

            plt.xlim(ximm[0].cpu(), ximm[-1].cpu())
            plt.ylim(zimm[-1].cpu(), zimm[0].cpu())
            fig.tight_layout()
            return hbi, hci, hbt, hct, bmax
        else:
            hbi.set_data(bimg.cpu())
            hci.set_data(cimg.cpu())
            hbt.set_text("SB: %.2f, CF: %.3f, PE: %.3f" % (sb_loss(c), cf_loss(c), pe_loss(c)))
            if CTRUE[sample] > 0:
                hct.set_text("Iteration %d: MAE %.2f" % (i, np.mean(np.abs(cimg.cpu() - CTRUE[sample]))))
            else:
                hct.set_text("Iteration %d: Mean value %.2f" % (i, torch.mean(cimg)))

        plt.savefig(f"scratch/{sample}.png")

    # Initialize figure
    print("Optimization loop...")
    handles = makeFigure(c, 0)

    # 优化循环初始化
    for i in tqdm(range(N_ITERS)):
        optimizer.zero_grad()
        loss_value = loss(c)
        loss_value.backward()
        optimizer.step()
        makeFigure(c, i + 1, handles)  # Update figure
        vobj.grab_frame()  # Add to video writer
    vobj.finish()  # Close video writer

    return c
