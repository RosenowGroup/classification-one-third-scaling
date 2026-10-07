import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.lines import Line2D
from numba import njit

try:
    from .util_iid_gaussian import calculate_theory_constants, colors, dark_colors
except ImportError:
    from util_iid_gaussian import calculate_theory_constants, colors, dark_colors


@njit
def run_sgd(T, J, eta, batch_size, N_steps, test_X, eval_steps, rng):
    K, N = J.shape
    num_test = test_X.shape[0]
    num_saves = len(eval_steps)

    alpha_hist = np.zeros(num_saves)
    D_hist = np.zeros((num_saves, K))
    Qeff_hist = np.zeros((num_saves, K))
    epsg_hist = np.zeros(num_saves)
    celoss_hist = np.zeros(num_saves)

    save_idx = 0
    scale = K / (K - 1.0)
    batch_gradient = np.zeros_like(J)

    for step in range(N_steps + 1):
        alpha_current = step / N

        if save_idx < num_saves and step == eval_steps[save_idx]:
            T_bar = np.zeros(N)
            J_bar = np.zeros(N)
            for k in range(K):
                T_bar += T[k]
                J_bar += J[k]
            T_bar /= K
            J_bar /= K

            for k in range(K):
                T_hat = np.sqrt(scale) * (T[k] - T_bar)
                J_hat = np.sqrt(scale) * (J[k] - J_bar)
                D_hist[save_idx, k] = np.dot(J_hat, T_hat) / N
                Qeff_hist[save_idx, k] = np.dot(J_hat, J_hat) / N

            alpha_hist[save_idx] = alpha_current
            errors = 0
            ce_sum = 0.0
            for i in range(num_test):
                xi_test = test_X[i]
                u = np.dot(T, xi_test) / np.sqrt(N)
                t = np.dot(J, xi_test) / np.sqrt(N)
                y = np.argmax(u)
                if y != np.argmax(t):
                    errors += 1

                max_t = np.max(t)
                exp_t = np.exp(t - max_t)
                p = exp_t / np.sum(exp_t)
                ce_sum -= np.log(p[y] + 1e-12)

            epsg_hist[save_idx] = errors / num_test
            celoss_hist[save_idx] = ce_sum / num_test
            save_idx += 1

        if step < N_steps:
            batch_gradient[:] = 0.0
            for _ in range(batch_size):
                xi = rng.normal(0.0, 1.0, N)
                u = np.dot(T, xi) / np.sqrt(N)
                t = np.dot(J, xi) / np.sqrt(N)
                y = np.argmax(u)

                max_t = np.max(t)
                exp_t = np.exp(t - max_t)
                p = exp_t / np.sum(exp_t)

                for k in range(K):
                    delta_ky = 1.0 if k == y else 0.0
                    batch_gradient[k] += (delta_ky - p[k]) * xi

            J += (eta / (batch_size * np.sqrt(N))) * batch_gradient

    return alpha_hist, D_hist, Qeff_hist, epsg_hist, celoss_hist


def get_run_filename(N, K, eta, batch_size, seed, alpha_max):
    return (
        f"run_N{N}_K{K}_eta_{eta:.4f}_batch_{batch_size}_"
        f"seed_{seed}_amax_{alpha_max}.npz"
    )


def generate_fig_base_name(
    N, K, eta, batch_size_list, seeds, alpha_max, d=10, alpha_min=1.0
):
    batch_str = "_".join(str(batch_size) for batch_size in batch_size_list)
    seed_str = "_".join(str(seed) for seed in seeds)
    return (
        f"N{N}_K{K}_eta_{eta:.2f}_batches_{batch_str}_seeds_{seed_str}_"
        f"d{d}_amin{alpha_min}_amax{alpha_max}"
    )


def plot_results(
    data,
    batch_size_list,
    eta,
    K,
    alpha_min,
    alpha_max,
    d,
    base_name,
    plot_idx,
    plot_kwargs,
    fig_dir,
    ylim_D,
    seeds,
    plot_theory=True,
):
    os.makedirs(fig_dir, exist_ok=True)

    if plot_kwargs is None:
        figsize = plt.rcParams["figure.figsize"]
        figsize_small = plt.rcParams["figure.figsize"]
        ax_dims = [0.2, 0.2, 0.75, 0.75]
        ax_dims_small = [0.2, 0.2, 0.75, 0.75]
        inset_dims = [0.2, 0.1, 0.35, 0.35]
    else:
        figsize = plot_kwargs.get("figsize", plt.rcParams["figure.figsize"])
        figsize_small = plot_kwargs.get("figsize_small", figsize)
        ax_dims = plot_kwargs.get("ax_dims", [0.2, 0.2, 0.75, 0.75])
        ax_dims_small = plot_kwargs.get("ax_dims_small", ax_dims)
        inset_dims = plot_kwargs.get("inset_dims", [0.2, 0.1, 0.4, 0.4])

    fig_D = plt.figure(figsize=figsize_small)
    ax_D = fig_D.add_axes(ax_dims_small)
    fig_Delta = plt.figure(figsize=figsize_small)
    ax_Delta = fig_Delta.add_axes(ax_dims_small)
    fig_Qeff = plt.figure(figsize=figsize_small)
    ax_Qeff = fig_Qeff.add_axes(ax_dims_small)
    fig_epsg = plt.figure(figsize=figsize)
    ax_epsg = fig_epsg.add_axes(ax_dims)
    ax_celoss = ax_epsg.inset_axes(inset_dims)

    for ax in [ax_D, ax_Delta, ax_Qeff, ax_epsg, ax_celoss]:
        ax.grid(False)

    if plot_theory:
        c_K, _, Gamma_K, _, _ = calculate_theory_constants(K)
        rng_theory = np.random.default_rng(42)

    alpha_hist = data["alpha"]
    valid_idx = alpha_hist > 0
    alpha_plot = alpha_hist[valid_idx]
    alpha_theory_b1 = None

    for idx, batch_size in enumerate(batch_size_list):
        color = colors[idx % len(colors)]
        dark_color = dark_colors[idx % len(dark_colors)]
        label = rf"$b={batch_size}$"

        D_all = np.array(
            [data[f"D_{batch_size}_seed_{seed}"] for seed in seeds]
        )
        Qeff_all = np.array(
            [data[f"Qeff_{batch_size}_seed_{seed}"] for seed in seeds]
        )
        epsg_all = np.array(
            [data[f"epsg_{batch_size}_seed_{seed}"] for seed in seeds]
        )
        celoss_all = np.array(
            [data[f"celoss_{batch_size}_seed_{seed}"] for seed in seeds]
        )
        Delta_all = Qeff_all[:, :, plot_idx] - D_all[:, :, plot_idx] ** 2

        ax_D.loglog(
            alpha_plot, D_all[0][valid_idx, plot_idx], color=color, label=label
        )
        ax_Delta.loglog(alpha_plot, Delta_all[0][valid_idx], color=color, label=label)
        ax_Qeff.loglog(
            alpha_plot, Qeff_all[0][valid_idx, plot_idx], color=color, label=label
        )
        ax_epsg.loglog(alpha_plot, epsg_all[0][valid_idx], color=color, label=label)
        ax_celoss.loglog(alpha_plot, celoss_all[0][valid_idx], color=color)

        if len(seeds) > 1:
            envelope_kwargs = dict(color=color, alpha=0.5, edgecolor="none")
            ax_D.fill_between(
                alpha_plot,
                np.min(D_all, axis=0)[valid_idx, plot_idx],
                np.max(D_all, axis=0)[valid_idx, plot_idx],
                **envelope_kwargs,
            )
            ax_Delta.fill_between(
                alpha_plot,
                np.min(Delta_all, axis=0)[valid_idx],
                np.max(Delta_all, axis=0)[valid_idx],
                **envelope_kwargs,
            )
            ax_Qeff.fill_between(
                alpha_plot,
                np.min(Qeff_all, axis=0)[valid_idx, plot_idx],
                np.max(Qeff_all, axis=0)[valid_idx, plot_idx],
                **envelope_kwargs,
            )
            ax_epsg.fill_between(
                alpha_plot,
                np.min(epsg_all, axis=0)[valid_idx],
                np.max(epsg_all, axis=0)[valid_idx],
                **envelope_kwargs,
            )
            ax_celoss.fill_between(
                alpha_plot,
                np.min(celoss_all, axis=0)[valid_idx],
                np.max(celoss_all, axis=0)[valid_idx],
                **envelope_kwargs,
            )

        if plot_theory:
            Z_samples = rng_theory.normal(0.0, 1.0, 500000)

            def B_func(Delta):
                inner = np.sqrt(Delta / 2.0) * Z_samples
                return np.mean(2.0 * np.log(2.0 * np.cosh(inner)) - 1.0)

            def root_eq(Delta):
                return 2.0 * Delta - (eta / batch_size) * B_func(Delta)

            Delta_star = opt.root_scalar(root_eq, bracket=[1e-6, 5.0]).root
            prefactor_D = (
                (3.0 * K * c_K / 2.0)
                * eta
                * (np.pi**2 / 6.0 + Delta_star)
            )
            c_eps = Gamma_K * np.sqrt(Delta_star) / prefactor_D ** (1.0 / 3.0)
            if alpha_theory_b1 is None:
                alpha_min_theory = c_eps**-6 / d
                if alpha_min_theory >= alpha_max:
                    alpha_min_theory = alpha_max / 10.0
                alpha_theory_b1 = np.logspace(
                    np.log10(alpha_min_theory), np.log10(alpha_max), 200
                )
            alpha_theory = alpha_theory_b1

            D_theory = (prefactor_D * alpha_theory) ** (1.0 / 3.0)
            Delta_theory = np.full_like(alpha_theory, Delta_star)
            Qeff_theory = D_theory**2 + Delta_star
            epsg_theory = c_eps * alpha_theory ** (-1.0 / 3.0)
            celoss_theory = (
                (K * (K - 1) * c_K / 2.0)
                * (np.pi**2 / 6.0 + Delta_star)
                / D_theory
            )

            theory_kwargs = dict(
                ls="--", dashes=[2, 1.8], linewidth=2.0, color=dark_color
            )
            ax_D.loglog(alpha_theory, D_theory, **theory_kwargs)
            ax_Delta.loglog(alpha_theory, Delta_theory, **theory_kwargs)
            ax_Qeff.loglog(alpha_theory, Qeff_theory, **theory_kwargs)
            ax_epsg.loglog(alpha_theory, epsg_theory, **theory_kwargs)
            ax_celoss.loglog(alpha_theory, celoss_theory, **theory_kwargs)

    if ylim_D is not None:
        ax_D.set_ylim(ylim_D)

    handles_proxy = []
    if plot_theory:
        handles_proxy.append(
            Line2D(
                [0],
                [0],
                ls="--",
                dashes=[1.5, 2],
                linewidth=1.5,
                color="grey",
                label="theory",
            )
        )

    ax_D.set_xlabel(r"Training Time $\alpha$")
    ax_D.set_ylabel(r"Overlap $D$", labelpad=0)
    ax_Delta.set_xlabel(r"Training Time $\alpha$")
    ax_Delta.set_ylabel(r"Residual Variance $\Delta$", labelpad=0)
    ax_Qeff.set_xlabel(r"Training Time $\alpha$")
    ax_Qeff.set_ylabel(r"Effective Overlap $Q_{\mathrm{eff}}$", labelpad=0)
    ax_epsg.set_xlabel(r"Training Time $\alpha$")
    ax_epsg.set_ylabel(r"Generalization Error $\epsilon_g$", labelpad=0)
    handles, _ = ax_epsg.get_legend_handles_labels()
    ax_epsg.legend(
        handles=handles + handles_proxy,
        loc="upper right",
        handlelength=1.0,
        labelspacing=0.1,
        frameon=False,
        borderpad=0.1,
        borderaxespad=0.4,
        handletextpad=0.5,
    )
    ax_celoss.set_ylabel(r"Test Loss $\mathcal{L}_g$", fontsize=7, labelpad=-2)
    ax_celoss.tick_params(axis="both", which="both", labelsize=6)

    fig_D.savefig(os.path.join(fig_dir, f"D_{base_name}.pdf"))
    fig_Delta.savefig(os.path.join(fig_dir, f"Delta_{base_name}.pdf"))
    fig_Qeff.savefig(os.path.join(fig_dir, f"Qeff_{base_name}.pdf"))
    fig_epsg.savefig(os.path.join(fig_dir, f"Epsg_{base_name}.pdf"))

    figs = {"epsg": fig_epsg, "Qeff": fig_Qeff, "Delta": fig_Delta, "D": fig_D}
    axes = {
        "epsg": ax_epsg,
        "Qeff": ax_Qeff,
        "Delta": ax_Delta,
        "D": ax_D,
        "celoss": ax_celoss,
    }
    print("Finished successfully. Figures saved.")
    plt.show()
    return figs, axes


def run_and_save_single_experiment(
    eta,
    batch_size,
    seed,
    N,
    K,
    alpha_max,
    num_test,
    data_dir,
    spacing="log",
    first_step=1,
    num_saves=200,
):
    os.makedirs(data_dir, exist_ok=True)
    filename = get_run_filename(N, K, eta, batch_size, seed, alpha_max)
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping simulation.")
        return dict(np.load(filepath))

    print(f"--- Simulating eta = {eta}, batch size = {batch_size} [Seed: {seed}] ---")
    N_steps = int(alpha_max * N)
    if spacing == "log":
        eval_steps = np.geomspace(first_step, N_steps, num_saves, dtype=int)
    else:
        eval_steps = np.linspace(first_step, N_steps, num_saves, dtype=int)
    eval_steps = np.unique(eval_steps)

    rng = np.random.default_rng(seed)
    T_raw = rng.normal(0.0, 1.0, (K, N))
    T = np.zeros((K, N))
    for i in range(K):
        v = T_raw[i]
        for j in range(i):
            v -= np.dot(T[j], v) / N * T[j]
        v = v / np.sqrt(np.dot(v, v) / N)
        T[i] = v

    test_X = rng.normal(0.0, 1.0, (num_test, N))
    J = rng.normal(0.0, 1.0, (K, N))
    alpha_hist, D_emp, Qeff_emp, epsg_emp, celoss_emp = run_sgd(
        T, J, eta, batch_size, N_steps, test_X, eval_steps, rng
    )

    data = {
        "alpha": alpha_hist,
        "D": D_emp,
        "Qeff": Qeff_emp,
        "epsg": epsg_emp,
        "celoss": celoss_emp,
        "eta": eta,
        "batch_size": batch_size,
        "seed": seed,
    }
    np.savez_compressed(filepath, **data)
    print(f"Data saved to {filepath}")
    return data


def run_parameter_sweep(
    batch_size_list=[1],
    eta=0.1,
    seeds=[42],
    num_test=100000,
    data_dir="data_K_class",
    fig_dir="figure_K_class",
    N=2000,
    K=3,
    alpha_max=200,
    plot_idx=0,
    spacing="log",
    first_step=1,
    num_saves=200,
    alpha_min=1.0,
    d=10,
    ylim_D=None,
    plot_kwargs=None,
    plot_theory=True,
):
    print(
        f"Running batch-size sweep: N={N}, K={K}, eta={eta}, "
        f"batch_sizes={batch_size_list}, alpha_max={alpha_max}, seeds={seeds}"
    )
    all_seeds_data = {}
    alpha_set = False

    for seed in seeds:
        for batch_size in batch_size_list:
            data = run_and_save_single_experiment(
                eta=eta,
                batch_size=batch_size,
                seed=seed,
                N=N,
                K=K,
                alpha_max=alpha_max,
                num_test=num_test,
                data_dir=data_dir,
                spacing=spacing,
                first_step=first_step,
                num_saves=num_saves,
            )
            if not alpha_set:
                all_seeds_data["alpha"] = data["alpha"]
                alpha_set = True

            all_seeds_data[f"D_{batch_size}_seed_{seed}"] = data["D"]
            all_seeds_data[f"Qeff_{batch_size}_seed_{seed}"] = data["Qeff"]
            all_seeds_data[f"epsg_{batch_size}_seed_{seed}"] = data["epsg"]
            all_seeds_data[f"celoss_{batch_size}_seed_{seed}"] = data["celoss"]

    base_name = generate_fig_base_name(
        N, K, eta, batch_size_list, seeds, alpha_max, d, alpha_min
    )
    plot_results(
        all_seeds_data,
        batch_size_list,
        eta,
        K,
        alpha_min,
        alpha_max,
        d,
        base_name,
        plot_idx,
        plot_kwargs,
        fig_dir,
        ylim_D,
        seeds,
        plot_theory,
    )
