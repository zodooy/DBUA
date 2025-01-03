import matplotlib.pyplot as plt


def imagesc(xc, y, img, dr, **kwargs):
    """MATLAB style imagesc"""
    dx = xc[1] - xc[0]
    dy = y[1] - y[0]
    ext = [xc[0] - dx / 2, xc[-1] + dx / 2, y[-1] + dy / 2, y[0] - dy / 2]
    im = plt.imshow(img, vmin=dr[0], vmax=dr[1], extent=ext, **kwargs)
    plt.colorbar()
    return im


def plot_errors_vs_sound_speeds(c0, dsb, dlc, dcf, dpe, sample):
    plt.clf()
    plt.plot(c0, dsb, label="Speckle Brightness")
    plt.plot(c0, dlc, label="Lag One Coherence")
    plt.plot(c0, dcf, label="Coherence Factor")
    # divided by 10 for visualization
    plt.plot(c0, dpe / 10, label="Phase Error")
    plt.grid()
    plt.xlabel("Global sound speed (m/s)")
    plt.ylabel("Loss function")
    plt.title(sample)
    plt.legend()
    plt.savefig(f"images/losses_{sample}.png", bbox_inches="tight", dpi=750)
    plt.clf()


def plot_loss(losses, sample):
    plt.clf()
    plt.plot(losses, label=sample)
    plt.grid()
    plt.xlabel("Iteration steps")
    plt.ylabel("Loss")
    plt.title(sample)
    plt.savefig(f"losses/optimize_{sample}.png", bbox_inches="tight", dpi=750)
    plt.clf()
