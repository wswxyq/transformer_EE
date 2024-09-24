"""Module for binning data and plotting histograms."""

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


def plot_xstat(x, y, stat1="mean", stat2="rms", name="xbinstat", **kwargs):
    # pylint: disable=too-many-locals
    """
    Plot the histogram with error bar
    This is a special plot, each error bar contains the mean and std of input y by default

    x, y: input array of data
    range: range of x, if None, use min(x) and max(x)
    bins: number of bins
    bin_y(stat1): statistic to plot as y
    bin_yerr(stat2): statistic to plot as error bar(vertical)
    bin_x: midpoint of bins

    optional:
    parameter       default val
    ----------      -----------
    bins            50
    range           (min(x), max(x))
    ext             "pdf"
    xlabel          ""
    ylabel          ""
    title           ""
    figsize         (8, 6)
    dpi             80
    plotgrid        True
    fmt             "none"
    color           "orange"
    outdir          "."

    --------------------------

    return: bin_x, bin_y, bin_yerr

    """

    _stat_var = {
        "mean": np.mean,
        "std": np.std,
        "rms": lambda x: np.sqrt(np.mean(x**2)),
        "stderr": lambda x: np.std(x) / np.sqrt(len(x)),
    }

    _range = kwargs.get("range", (min(x), max(x)))

    bin_y, _bin_edges, _binnumber = stats.binned_statistic(
        x, y, statistic=_stat_var[stat1], bins=kwargs.get("bins", 50), range=_range
    )
    bin_yerr, _bin_edges, _binnumber = stats.binned_statistic(
        x, y, statistic=_stat_var[stat2], bins=kwargs.get("bins", 50), range=_range
    )
    bin_x = (_bin_edges[:-1] + _bin_edges[1:]) / 2

    _fig, _ax = plt.subplots(
        1, 1, figsize=kwargs.get("figsize", (8, 6)), dpi=kwargs.get("dpi", 80)
    )
    errb = _ax.errorbar(
        bin_x,
        bin_y,
        yerr=bin_yerr,
        xerr=(_bin_edges[1:] - _bin_edges[:-1]) / 2,
        fmt=kwargs.get("fmt", "none"),
        ecolor=kwargs.get("color", "orange"),
        label=f"{stat1.upper()} : {_stat_var[stat1](y):.1%}, {stat2.upper()} : {_stat_var[stat2](y):.1%}",
    )
    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(
        kwargs.get("title", stat1.upper() + " + E[" + stat2.upper() + " ]"), fontsize=16
    )
    _ax.hlines(0, _range[0], _range[1], linestyles="dashed", color="green")
    _ax.set_xlim(_range[0], _range[1])

    if kwargs.get("plotgrid", True):
        _ax.grid(
            visible=True, which="major", color="#666666", linestyle="--", alpha=0.8
        )
        _ax.minorticks_on()
        _ax.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    _ax.legend(loc="best", handles=[errb])
    plt.savefig(
        os.path.join(
            kwargs.get("outdir", "."),
            name + "_" + stat1 + "_" + stat2 + "." + kwargs.get("ext", "pdf"),
        )
    )

    return {"bin_x": bin_x, "bin_y": bin_y, "bin_yerr": bin_yerr}


def plot_y_hist(x, name="yhist", **kwargs):
    # pylint: disable=too-many-locals
    # pylint: disable=unbalanced-tuple-unpacking
    """
    Make a histogram of x and a gaussian fit

    optional:
    parameter       default val
    ----------      -----------
    bins            50
    range           (min(x), max(x))
    ext             "pdf"
    xlabel          ""
    ylabel          ""
    title           ""
    figsize         (8, 6)
    dpi             80
    outdir          "."

    --------------------------

    return: mu, sigma, rms, a
    """

    def gaussian_like(_x, _mu, _sigma, _a):
        return (
            _a
            * (1 / (_sigma * np.sqrt(2 * np.pi)))
            * np.exp(-(((_x - _mu) / _sigma) ** 2) / 2)
        )

    stat_mu, stat_sigma = np.mean(x), np.std(x)
    stat_rms = np.sqrt(np.mean(x**2))

    _range = kwargs.get("range", (-1, 1))

    _fig, _ax = plt.subplots(
        1, 1, figsize=kwargs.get("figsize", (8, 6)), dpi=kwargs.get("dpi", 80)
    )
    _n, _b, _p = _ax.hist(
        x,
        bins=kwargs.get("bins", 50),
        range=_range,
        weights=kwargs.get("weights", None),
        histtype="step",
    )

    popt, _pcov = curve_fit(gaussian_like, (_b[1:] + _b[:-1]) / 2, _n)

    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(
        kwargs.get("ylabel", f"NumEvents / {(_b[1] - _b[0]):.3f}\n"), fontsize=14
    )
    _ax.set_title(kwargs.get("title", ""), fontsize=18)
    _ax.text(
        0.95,
        0.95,
        f"NumEvents : {len(x):.3e}\n\
            STAT:\n\
            MEAN : {stat_mu:.1%}\n\
            RMS : {stat_rms:.1%}\n\
            Std : {stat_sigma:.1%}\n\
            \n\
            Gaussian:\n\
            MEAN : {popt[0]:.1%}\n\
            Std : {popt[1]:.1%}\n",
        transform=_ax.transAxes,
        ha="right",
        va="top",
    )

    _xplot = np.linspace(_range[0], _range[1], np.max(kwargs.get("bins", 50) * 4, 200))
    _ax.plot(_xplot, gaussian_like(_xplot, *popt))
    _ax.set_xlim(_range[0], _range[1])
    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "pdf"))
    )
    return {
        "stat": {
            "mu": stat_mu,
            "sigma": stat_sigma,
            "rms": stat_rms,
        },
        "fit": {
            "fit_mu": popt[0],
            "fit_sigma": popt[1],
        },
    }


def plot_2d_hist_count(x, y, name="hist2D", **kwargs):
    """
    Make a 2D histogram of x and y

    optional:
    parameter       default val
    ----------      -----------
    bins            50
    xbins           bins
    ybins           bins
    xrange          (0, 6)
    yrange          (0, 6)
    ext             "pdf"
    xlabel          ""
    ylabel          ""
    title           ""
    figsize         (8, 8)
    dpi             80
    outdir          "."

    --------------------------

    return: mu, sigma, rms, a
    """

    _bins = kwargs.get("bins", 50)
    _xbins = kwargs.get("xbins", _bins)
    _ybins = kwargs.get("ybins", _bins)
    _xrange = kwargs.get("xrange", (0, 6))
    _yrange = kwargs.get("yrange", (0, 6))

    H, xedges, yedges = np.histogram2d(
        x, y, bins=(_xbins, _ybins), range=(_xrange, _yrange)
    )

    H = H.T
    _fig, _ax = plt.subplots(
        1, 1, figsize=kwargs.get("figsize", (8, 8)), dpi=kwargs.get("dpi", 80)
    )
    X, Y = np.meshgrid(xedges, yedges)
    im = _ax.pcolormesh(X, Y, H, edgecolors="face")
    _ax.set_xlabel(kwargs.get("xlabel", ""), fontsize=14)
    _ax.set_ylabel(kwargs.get("ylabel", ""), fontsize=14)
    _ax.set_title(kwargs.get("title", ""), fontsize=16)

    cax = _fig.add_axes(
        [
            _ax.get_position().x1 + 0.01,
            _ax.get_position().y0,
            0.02,
            _ax.get_position().height,
        ]
    )
    plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)

    plt.savefig(
        os.path.join(kwargs.get("outdir", "."), name + "." + kwargs.get("ext", "pdf"))
    )
