import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm

# plt.rcParams["text.usetex"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = ["cm"]
plt.rcParams["font.size"] = 12


def create_subplots(nrows) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Create subplots with specific size.
    """

    # Subplot size
    width, height = 5, 2  # inches

    # Figure margins
    left, right = 0.7, 0.7  # inches
    top, bottom = 0.3, 0.6  # inches

    # Vertical space before subplots
    vspace = 0.6  # inches

    # Create figure with margins
    fig_height = nrows * height + top + bottom  # inches
    fig_width = width + left + right  # inches
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Create axes with shared x
    axes = []
    for i in range(nrows):
        ax_left = left / fig_width  # fraction
        ax_bottom = (fig_height - top - (i + 1) * height) / fig_height  # fraction
        ax_width = (fig_width - right - left) / fig_width  # fraction
        ax_height = (height - vspace) / fig_height  # fraction
        sharex = axes[0] if axes else None
        ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height], sharex=sharex)
        if i < nrows - 1:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
    return fig, axes


def save_figure(fig: plt.Figure, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path)


def combine_legends(*axes: plt.Axes, loc="upper right"):
    all_lines, all_labels = [], []
    for ax in axes:
        lines, labels = ax.get_legend_handles_labels()
        all_lines.extend(lines)
        all_labels.extend(labels)
        ax.legend().remove()
    axes[-1].legend(all_lines, all_labels, loc=loc)


def twinx(plot):
    def _plot(ax: plt.Axes, *args, **kwargs):
        ax1 = ax.twinx()
        plot(ax1, *args, **kwargs)
        ax1.grid(False)
        ax1.set_title("")
        combine_legends(ax, ax1)
        return ax1

    return _plot


def plot_measurements(ax: plt.Axes, t, y):
    ax.plot(t, y, color="black", label=r"$y$")
    ax.set_title("Observed SC Measurements")
    ax.set_ylabel(r"SC Measurement ($\mu S$)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_mpp(ax: plt.Axes, t, x_pred, r, r0, r1, vr):
    r_pred = r0 + r1 * x_pred
    lcl_r = norm.ppf(0.025, r_pred, np.sqrt(vr))
    ucl_r = norm.ppf(0.975, r_pred, np.sqrt(vr))

    ax.plot(t, r, color="black", label=r"$r_{obs}$")
    ax.plot(t, r_pred, color="red", label=r"$r_{pred}$")
    ax.fill_between(t, lcl_r, ucl_r, color="red", alpha=0.3, label=r"95\% CI")
    ax.set_title("ANS Activations" + "\n" + rf"$(\gamma_0={r0:.3f}, \gamma_1={r1:.3f}, \sigma^2_r={vr:.3f})$")
    ax.set_ylabel(r"ANS Activation ($\mu S/s$)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_cont(ax: plt.Axes, t, x_pred, s, s0, s1, vs):
    s_pred = s0 + s1 * x_pred
    lcl_s = norm.ppf(0.025, s_pred, np.sqrt(vs))
    ucl_s = norm.ppf(0.975, s_pred, np.sqrt(vs))

    ax.plot(t, s, color="black", label=r"$s_{obs}$")
    ax.plot(t, s_pred, color="red", label=r"$s_{pred}$")
    ax.fill_between(t, lcl_s, ucl_s, color="red", alpha=0.3, label=r"95\% CI")
    ax.set_title("Tonic SC" + "\n" + rf"$(\delta_0={s0:.3f}, \delta_1={s1:.3f}, \sigma^2_s={vs:.3f})$")
    ax.set_ylabel(r"Tonic SC ($\mu S$)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_states(ax: plt.Axes, t, x_pred, v_pred, vx=None, x_true=None):
    lcl_x = norm.ppf(0.025, x_pred, np.sqrt(v_pred))
    ucl_x = norm.ppf(0.975, x_pred, np.sqrt(v_pred))

    if x_true is not None:
        ax.plot(t, x_true, "black", label=r"$x_{true}$")
    ax.plot(t, x_pred, "red", label=r"$x_{pred}$")
    ax.fill_between(t, lcl_x, ucl_x, color="red", alpha=0.3, label=r"95\% CI")
    title = "Estimated Arousal States"
    if vx is not None:
        title += "\n" + rf"$(\sigma^2_x={vx:.3f})$"
    ax.set_title(title)
    ax.set_ylabel(r"Arousal State (a.u.)")
    ax.grid()
    ax.legend(loc="upper right")


def plot_time(ax: plt.Axes, t):
    ax.set_xlim(left=t[0], right=t[-1])
    ax.set_xlabel(r"Time ($s$)")


def plot_title(ax: plt.Axes, title=None):
    ax_title = ax.get_title()
    if ax_title == "" and title is None:
        _title = ""
    elif ax_title == "" and title is not None:
        _title = title
    elif ax_title != "" and title is None:
        _title = ax_title
    else:
        _title = title + "\n\n" + ax_title
    ax.set_title(_title)


def plot_results(
    t, y, x_pred, v_pred, vx, r, r0, r1, vr, s=None, s0=None, s1=None, vs=None, x_true=None, title=None, save_path=None
):
    fig, axes = create_subplots(3 + (s is not None))

    plot_measurements(axes[0], t, y)
    plot_states(axes[1], t, x_pred, v_pred, vx, x_true)
    plot_mpp(axes[2], t, x_pred, r, r0, r1, vr)
    if s is not None:
        plot_cont(axes[3], t, x_pred, s, s0, s1, vs)
    plot_time(axes[-1], t)
    plot_title(axes[0], title=title)

    # Remove legends
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.legend().remove()

    if save_path is not None:
        save_figure(fig, save_path)
        plt.close(fig)
    else:
        return fig, axes


def plot_results_states_only(t, x_pred, v_pred, title=None, save_path=None):
    fig, axes = create_subplots(1)
    ax = axes[0]

    plot_states(ax, t, x_pred, v_pred)
    plot_time(ax, t)
    plot_title(ax, title=title)

    # Remove legends
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.legend().remove()

    if save_path is not None:
        save_figure(fig, save_path)
        plt.close(fig)
    else:
        return fig, axes


def main(exp_name, decoder="mpp_cont"):
    load_path = f"results/log/{exp_name}/{{subject}}_{{phase}}.mat"
    save_path = f"results/paper/plots/{decoder}"

    os.makedirs(save_path, exist_ok=True)

    # Dictionary for display
    phase_names = {
        "cond": "Conditioning",
        "ext": "Extinction",
        "recall": "Recall",
    }

    # Get the list of subjects and phases
    dataset = pd.read_csv("dataset.csv")
    dataset = dataset.groupby(["subject", "phase"])

    # Main loop
    for (subject, phase), data in dataset:
        group = data["group"].iloc[0]

        # Load result
        result = loadmat(load_path.format(subject=subject, phase=phase))

        t = result["t"].squeeze(0)
        y = result["y"].squeeze(0)
        x_pred = result["x_smth"].squeeze(0)
        v_pred = result["v_smth"].squeeze(0)
        vx = result["vx"].item()
        r = result["r"].squeeze(0)
        r0 = result["r0"].item()
        r1 = result["r1"].item()
        vr = result["vr"].item()
        if "s0" in result:  # if mpp_cont
            s = result["s"].squeeze(0)
            s0 = result["s0"].item()
            s1 = result["s1"].item()
            vs = result["vs"].item()
        else:  # if mpp
            s, s0, s1, vs = None, None, None, None

        # Plot
        phase_name = phase_names[phase]
        title = f"Subject {subject}, Group: \\textbf{{{group}}}, Phase: \\textbf{{{phase_name}}}"
        filename = f"{subject}_{phase}"

        _save_path = os.path.join(save_path, f"{filename}.pdf")
        plot_results(t, y, x_pred, v_pred, vx, r, r0, r1, vr, s, s0, s1, vs, title=title, save_path=_save_path)

        _save_path = os.path.join(save_path, f"{filename}_states.pdf")
        plot_results_states_only(t, x_pred, v_pred, title=title, save_path=_save_path)


if __name__ == "__main__":
    main("mpp_cont_best", decoder="mpp_cont")
    main("mpp_best", decoder="mpp")
