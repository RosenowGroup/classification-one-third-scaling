import os
import numpy as np
import scipy.integrate as spi
import scipy.stats as stats
import scipy.optimize as opt
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================
# 1. Plot Style Configuration
# ==========================================
try:
    plt.style.use('plt_style.pstyle')
except:
    pass # Fails gracefully if the user does not have this specific file locally
    
plt.rcParams["font.size"] = 7.8        # tick label
plt.rcParams["figure.titlesize"] = 9   # title
plt.rcParams["legend.fontsize"] = 7    # legend
plt.rcParams["axes.titlesize"] = 9     # axes title
plt.rcParams["axes.labelsize"] = 8.8   # axes label
plt.rcParams['figure.figsize'] = [3.2, 2.4]
plt.rcParams["mathtext.fontset"] = "cm"

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#8c564b', '#bcbd22', '#adfc03']
light_colors = ["#a0cff8", "#99ffbe", "#ffd599", "#ffbfb3"]
dark_colors = ["#003A6D", "#007029", "#C57300", "#BF2000"]

# ==========================================
# 2. Numba-Compiled Microscopic Dynamics
# ==========================================

@njit
def run_sgd(T, J, eta_0, alpha_0, gamma, N_steps, test_X, eval_steps, rng):
    """
    Runs online SGD with Numba for extreme performance.
    Uses a scheduled learning rate: eta = eta_0 * (1 + alpha/alpha_0)**(-gamma).
    (If gamma=0, this is just fixed learning rate eta_0).
    """
    K = J.shape[0]
    N = J.shape[1]
    num_test = test_X.shape[0]
    num_saves = len(eval_steps)
    
    alpha_hist = np.zeros(num_saves)
    D_hist = np.zeros((num_saves, K))
    Qeff_hist = np.zeros((num_saves, K))
    epsg_hist = np.zeros(num_saves)
    celoss_hist = np.zeros(num_saves)
    
    save_idx = 0
    scale = K / (K - 1.0)
    
    for step in range(N_steps + 1):
        
        alpha_current = step / N

        # --- Periodic Saving & Order Parameter Tracking ---
        if save_idx < num_saves and step == eval_steps[save_idx]:
            T_bar = np.zeros(N)
            J_bar = np.zeros(N)
            for k in range(K):
                T_bar += T[k]
                J_bar += J[k]
            T_bar /= K
            J_bar /= K
            
            for k in range(K):
                T_tilde = T[k] - T_bar
                J_tilde = J[k] - J_bar
                
                T_hat = np.sqrt(scale) * T_tilde
                J_hat = np.sqrt(scale) * J_tilde
                
                D_hist[save_idx, k] = np.dot(J_hat, T_hat) / N
                Qeff_hist[save_idx, k] = np.dot(J_hat, J_hat) / N
                
            alpha_hist[save_idx] = alpha_current
            
            # Generalization & Cross-Entropy Error (over test set)
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
            
        # --- Microscopic SGD Update ---
        if step < N_steps:
            eta_step = eta_0 * (1.0 + alpha_current / alpha_0)**(-gamma)

            xi = rng.normal(0.0, 1.0, N)
            u = np.dot(T, xi) / np.sqrt(N)
            t = np.dot(J, xi) / np.sqrt(N)
            
            y = np.argmax(u)
            
            max_t = np.max(t)
            exp_t = np.exp(t - max_t)
            p = exp_t / np.sum(exp_t)
            
            for k in range(K):
                delta_ky = 1.0 if k == y else 0.0
                J[k] = J[k] + (eta_step / np.sqrt(N)) * (delta_ky - p[k]) * xi
            
    return alpha_hist, D_hist, Qeff_hist, epsg_hist, celoss_hist


# ==========================================
# 3. Utilities & Theory Functions
# ==========================================

def get_run_filename(N, K, eta_0, gamma, alpha_0, seed, alpha_max):
    """Generates a highly specific filename for an isolated run."""
    return f"run_N{N}_K{K}_eta_{eta_0:.4f}_gamma_{gamma:.4f}_a0_{alpha_0}_seed_{seed}_amax_{alpha_max}.npz"

def generate_fig_base_name(N, K, eta_list, gamma_list, alpha_0, seeds, alpha_max, d=10, alpha_min=1.0):
    """Generates the base string for the output figure covering multiple runs."""
    eta_str = "_".join([f"{lr:.2f}" for lr in eta_list])
    gamma_str = "_".join([f"{g:.2f}" for g in gamma_list])
    seed_str = "_".join([str(s) for s in seeds])
    return f"N{N}_K{K}_etas_{eta_str}_gammas_{gamma_str}_a0{alpha_0}_seeds_{seed_str}_d{d}_amin{alpha_min}_amax{alpha_max}"

def calculate_theory_constants(K):
    def cK_integrand(s):
        return (stats.norm.pdf(s)**2) * (stats.norm.cdf(s)**(K-2))
    c_K, _ = spi.quad(cK_integrand, -15, 15)
    
    kappa = (2.0 * np.log(2.0) - 1.0) / 2.0
    Gamma_K = K * (K - 1.0) * c_K / np.sqrt(np.pi)
    A_K = Gamma_K * np.sqrt(kappa) * (4.0 / (K * c_K * np.pi**2))**(1.0/3.0)
    
    Xi_K = (np.pi**2 / 12.0) * K * (K - 1.0) * c_K
    B_K = Xi_K * (4.0 / (K * c_K * np.pi**2))**(1.0/3.0)
    
    return c_K, kappa, Gamma_K, A_K, B_K


# ==========================================
# 4. Universal Plotting Engine
# ==========================================

def plot_results(data, is_gamma_sweep, sweep_vals, fixed_val, N, K, alpha_0, alpha_min, alpha_max, 
                 d, base_name, plot_idx, plot_kwargs, fig_dir, ylim_D, seeds, plot_theory=True):
    os.makedirs(fig_dir, exist_ok=True)

    # Modified plot_kwargs extraction to support small figures for D, Qeff, and Delta
    if plot_kwargs is None:
        figsize = plt.rcParams['figure.figsize']
        figsize_small = plt.rcParams['figure.figsize']
        ax_dims = [0.2, 0.2, 0.75, 0.75]
        ax_dims_small = [0.2, 0.2, 0.75, 0.75]
        inset_dims = [0.2, 0.1, 0.35, 0.35]
    else:
        figsize = plot_kwargs.get('figsize', plt.rcParams['figure.figsize'])
        figsize_small = plot_kwargs.get('figsize_small', figsize)
        ax_dims = plot_kwargs.get('ax_dims', [0.2, 0.2, 0.75, 0.75])
        ax_dims_small = plot_kwargs.get('ax_dims_small', ax_dims)
        inset_dims = plot_kwargs.get('inset_dims', [0.2, 0.1, 0.4, 0.4])

    # Apply small dimensions to D, Delta, and Qeff
    fig_D = plt.figure(figsize=figsize_small)
    ax_D = fig_D.add_axes(ax_dims_small)
    
    fig_Delta = plt.figure(figsize=figsize_small)
    ax_Delta = fig_Delta.add_axes(ax_dims_small)
    
    fig_Qeff = plt.figure(figsize=figsize_small)
    ax_Qeff = fig_Qeff.add_axes(ax_dims_small)
    
    # Apply regular dimensions to EPSG and Cross Entropy
    fig_epsg = plt.figure(figsize=figsize)
    ax_epsg = fig_epsg.add_axes(ax_dims)
    ax_celoss = ax_epsg.inset_axes(inset_dims)

    axes = [ax_D, ax_Delta, ax_Qeff, ax_epsg, ax_celoss]
    for ax in axes:
        ax.grid(False) 

    if plot_theory:
        c_K, kappa, Gamma_K, A_K, B_K = calculate_theory_constants(K)
        rng_theory = np.random.default_rng(42)
    
    alpha_hist = data["alpha"]
    valid_idx = alpha_hist > 0
    alpha_plot = alpha_hist[valid_idx]

    for idx, val in enumerate(sweep_vals):
        eta_0 = fixed_val if is_gamma_sweep else val
        gamma = val if is_gamma_sweep else fixed_val
        
        color = colors[idx % len(colors)]
        dark_color = dark_colors[idx % len(dark_colors)]
        label_str = rf"$\gamma={gamma}$" if is_gamma_sweep else rf"$\eta={eta_0}$"

        # Collect multi-seed empirical data
        D_all = np.array([data[f"D_{val}_seed_{s}"] for s in seeds])
        Qeff_all = np.array([data[f"Qeff_{val}_seed_{s}"] for s in seeds])
        epsg_all = np.array([data[f"epsg_{val}_seed_{s}"] for s in seeds])
        celoss_all = np.array([data[f"celoss_{val}_seed_{s}"] for s in seeds])
        
        Delta_all = Qeff_all[:, :, plot_idx] - D_all[:, :, plot_idx]**2

        # Empirical Data Plotting (Plot Seed 0 as primary line)
        ax_D.loglog(alpha_plot, D_all[0][valid_idx, plot_idx], color=color, label=label_str)
        ax_Delta.loglog(alpha_plot, Delta_all[0][valid_idx], color=color, label=label_str)
        ax_Qeff.loglog(alpha_plot, Qeff_all[0][valid_idx, plot_idx], color=color, label=label_str)
        ax_epsg.loglog(alpha_plot, epsg_all[0][valid_idx], color=color, label=label_str)
        ax_celoss.loglog(alpha_plot, celoss_all[0][valid_idx], color=color)

        # Draw Envelopes if multi-seed is used
        if len(seeds) > 1:
            envelope_kwargs = dict(color=color, alpha=0.5, edgecolor="none")
            ax_D.fill_between(alpha_plot, np.min(D_all, axis=0)[valid_idx, plot_idx], np.max(D_all, axis=0)[valid_idx, plot_idx], **envelope_kwargs)
            ax_Delta.fill_between(alpha_plot, np.min(Delta_all, axis=0)[valid_idx], np.max(Delta_all, axis=0)[valid_idx], **envelope_kwargs)
            ax_Qeff.fill_between(alpha_plot, np.min(Qeff_all, axis=0)[valid_idx, plot_idx], np.max(Qeff_all, axis=0)[valid_idx, plot_idx], **envelope_kwargs)
            ax_epsg.fill_between(alpha_plot, np.min(epsg_all, axis=0)[valid_idx], np.max(epsg_all, axis=0)[valid_idx], **envelope_kwargs)
            ax_celoss.fill_between(alpha_plot, np.min(celoss_all, axis=0)[valid_idx], np.max(celoss_all, axis=0)[valid_idx], **envelope_kwargs)

        if plot_theory:
            if gamma == 0.0:
                Z_samples = rng_theory.normal(0.0, 1.0, 500000)
                def B_func(Delta):
                    inner = np.sqrt(Delta / 2.0) * Z_samples
                    val = 2.0 * np.log(2.0 * np.cosh(inner)) - 1.0
                    return np.mean(val)
                    
                def root_eq(Delta):
                    return 2 * Delta - eta_0 * B_func(Delta)
                    
                res = opt.root_scalar(root_eq, bracket=[1e-6, 5.0])
                Delta_star = res.root
                
                prefactor_D = (3 * K * c_K / 2) * eta_0 * (np.pi**2 / 6 + Delta_star)
                c_eps = Gamma_K * np.sqrt(Delta_star) / (prefactor_D ** (1.0 / 3.0))

                # If eta is varied, use the more involved calculation for alpha_min
                if not is_gamma_sweep:
                    alpha_min_theory = (c_eps**-6) / d
                    if alpha_min_theory >= alpha_max:
                        alpha_min_theory = alpha_max / 10.0
                    alpha_theory = np.logspace(np.log10(alpha_min_theory), np.log10(alpha_max), 200)
                else:
                    alpha_theory = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 200)

                D_theory = (prefactor_D * alpha_theory) ** (1.0 / 3.0)
                Delta_theory = np.full_like(alpha_theory, Delta_star)
                Qeff_theory = D_theory**2 + Delta_star
                
                epsg_theory = c_eps * alpha_theory ** (-1.0 / 3.0)
                
                B_ce = (K * (K - 1) * c_K / 2) * ((2 / (3 * K * c_K * eta_0))**(1.0/3.0)) * ((np.pi**2 / 6 + Delta_star)**(2.0/3.0))
                celoss_theory = B_ce * alpha_theory ** (-1.0 / 3.0)
                
            else:
                alpha_theory = np.logspace(np.log10(alpha_min), np.log10(alpha_max), 200)
                eta_theory = eta_0 * (1.0 + alpha_theory / alpha_0)**(-gamma)
                
                if gamma == 1.0:
                    H_theory = eta_0 * alpha_0 * np.log(1.0 + alpha_theory / alpha_0)
                else:
                    H_theory = (eta_0 * alpha_0 / (1.0 - gamma)) * ((1.0 + alpha_theory / alpha_0)**(1.0 - gamma) - 1.0)
                    
                D_theory = ((K * c_K * np.pi**2 / 4.0) * H_theory)**(1.0 / 3.0)
                Delta_theory = kappa * eta_theory
                Qeff_theory = D_theory**2 + Delta_theory
                epsg_theory = A_K * np.sqrt(eta_theory) / (H_theory**(1.0 / 3.0))
                celoss_theory = B_K * (H_theory**(-1.0 / 3.0))
            
            # Theory Plotting
            theory_kwargs = dict(ls="--", dashes=[2, 1.8], linewidth=2.0, color=dark_color)
            ax_D.loglog(alpha_theory, D_theory, **theory_kwargs)
            ax_Delta.loglog(alpha_theory, Delta_theory, **theory_kwargs)
            ax_Qeff.loglog(alpha_theory, Qeff_theory, **theory_kwargs)
            ax_epsg.loglog(alpha_theory, epsg_theory, **theory_kwargs)
            ax_celoss.loglog(alpha_theory, celoss_theory, **theory_kwargs)

    # Apply limits
    if ylim_D is not None:
        ax_D.set_ylim(ylim_D)

    # Creating Proxy Theory Artist for the legends with short dashes exclusively 
    handles_proxy = []
    if plot_theory:
        theory_proxy = Line2D([0], [0], ls='--', dashes=[1.5, 2], linewidth=1.5, color='grey', label='theory')
        handles_proxy.append(theory_proxy)

    # Formatting 
    ax_D.set_xlabel(r"Training Time $\alpha$")
    ax_D.set_ylabel(r"Overlap $D$", labelpad=0)
    handles, labels = ax_D.get_legend_handles_labels()
    #ax_D.legend(handles=handles + handles_proxy, loc='lower right', handlelength=1.0)
    
    ax_Delta.set_xlabel(r"Training Time $\alpha$")
    ax_Delta.set_ylabel(r"Residual Variance $\Delta$", labelpad=0)
    handles, labels = ax_Delta.get_legend_handles_labels()
    #ax_Delta.legend(handles=handles + handles_proxy, loc='lower left', handlelength=1.0)
    
    ax_Qeff.set_xlabel(r"Training Time $\alpha$")
    ax_Qeff.set_ylabel(r"Effective Overlap $Q_{\mathrm{eff}}$", labelpad=0)
    handles, labels = ax_Qeff.get_legend_handles_labels()
    #ax_Qeff.legend(handles=handles + handles_proxy, loc='upper left', handlelength=1.0)
    
    ax_epsg.set_xlabel(r"Training Time $\alpha$")
    ax_epsg.set_ylabel(r"Generalization Error $\epsilon_g$", labelpad=0)
    handles, labels = ax_epsg.get_legend_handles_labels()
    ax_epsg.legend(handles=handles + handles_proxy, loc='upper right', handlelength=1.0,
                   labelspacing=0.1, frameon=False, borderpad=0.1, borderaxespad=0.4,
                   handletextpad=0.5)
    
    ax_celoss.set_ylabel(r"Test Loss $\mathcal{L}_g$", fontsize=7, labelpad=-2)
    ax_celoss.tick_params(axis='both', which='both', labelsize=6)

    fig_D.savefig(os.path.join(fig_dir, f"D_{base_name}.pdf"))
    fig_Delta.savefig(os.path.join(fig_dir, f"Delta_{base_name}.pdf"))
    fig_Qeff.savefig(os.path.join(fig_dir, f"Qeff_{base_name}.pdf"))
    fig_epsg.savefig(os.path.join(fig_dir, f"Epsg_{base_name}.pdf"))

    
    figs={"epsg":fig_epsg, "Qeff": fig_Qeff, "Delta":fig_Delta, "D":fig_D}
    axes={"epsg":ax_epsg, "Qeff": ax_Qeff, "Delta":ax_Delta, "D":ax_D, "celoss":ax_celoss}

    print("Finished successfully. Figures saved.")
    plt.show()

    return (figs, axes)


# ==========================================
# 5. Main Runners (Simulate / Load)
# ==========================================

def run_and_save_single_experiment(eta_0, gamma, seed, N, K, alpha_0, alpha_max, num_test, data_dir, spacing='log', first_step=1, num_saves=200):
    """
    Handles initialization, SGD execution, and file saving for ONE specific hyperparameter configuration.
    """
    os.makedirs(data_dir, exist_ok=True)
    filename = get_run_filename(N, K, eta_0, gamma, alpha_0, seed, alpha_max)
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print(f"File {filename} already exists. Skipping simulation.")
        return dict(np.load(filepath))

    print(f"--- Simulating eta_0 = {eta_0}, gamma = {gamma} [Seed: {seed}] ---")

    N_steps = int(alpha_max * N)
    if spacing == 'log':
        eval_steps = np.geomspace(first_step, N_steps, num_saves, dtype=int)
    else:
        eval_steps = np.linspace(first_step, N_steps, num_saves, dtype=int)
    eval_steps = np.unique(eval_steps) 

    # Isolate RNG state strictly to this exact run to ensure perfect initial condition parity
    rng = np.random.default_rng(seed)

    # Initialize Teacher Vectors
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
        T, J, eta_0, alpha_0, gamma, N_steps, test_X, eval_steps, rng
    )
    
    data_dict = {
        "alpha": eval_steps / N,
        "D": D_emp,
        "Qeff": Qeff_emp,
        "epsg": epsg_emp,
        "celoss": celoss_emp,
        "eta_0": eta_0,
        "gamma": gamma,
        "seed": seed
    }

    np.savez_compressed(filepath, **data_dict)
    print(f"Data saved to {filepath}")
    
    return data_dict

def run_parameter_sweep(eta_list=[1.0], gamma_list=[0.0], seeds=[42], num_test=100000, data_dir="data_K_class", fig_dir="figure_K_class",
                        alpha_0=1.0, N=2000, K=3, alpha_max=200, plot_idx=0, 
                        spacing='log', first_step=1, num_saves=200, alpha_min=1.0, d=10, 
                        ylim_D=None, plot_kwargs=None, plot_theory=True):
                        
    is_gamma_sweep = len(gamma_list) > 1
    sweep_vals = gamma_list if is_gamma_sweep else eta_list
    fixed_val = eta_list[0] if is_gamma_sweep else gamma_list[0]

    print(f"Running sweep: N={N}, K={K}, param_swept={'gamma' if is_gamma_sweep else 'eta'}, alpha_max={alpha_max}, seeds={seeds}")
    
    all_seeds_data = {}
    alpha_set = False

    for seed in seeds:
        for val in sweep_vals:
            eta_0 = fixed_val if is_gamma_sweep else val
            gamma = val if is_gamma_sweep else fixed_val
            
            data = run_and_save_single_experiment(
                eta_0=eta_0, gamma=gamma, seed=seed, N=N, K=K, alpha_0=alpha_0, 
                alpha_max=alpha_max, num_test=num_test, data_dir=data_dir, 
                spacing=spacing, first_step=first_step, num_saves=num_saves
            )

            # Accumulate into expected plotting dictionary format
            if not alpha_set:
                all_seeds_data["alpha"] = data["alpha"]
                alpha_set = True

            all_seeds_data[f"D_{val}_seed_{seed}"] = data["D"]
            all_seeds_data[f"Qeff_{val}_seed_{seed}"] = data["Qeff"]
            all_seeds_data[f"epsg_{val}_seed_{seed}"] = data["epsg"]
            all_seeds_data[f"celoss_{val}_seed_{seed}"] = data["celoss"]

    fig_base_name = generate_fig_base_name(N, K, eta_list, gamma_list, alpha_0, seeds, alpha_max, d, alpha_min)
    plot_results(all_seeds_data, is_gamma_sweep, sweep_vals, fixed_val, N, K, alpha_0, alpha_min, alpha_max, 
                 d, fig_base_name, plot_idx, plot_kwargs, fig_dir, ylim_D, seeds, plot_theory)

def plot_parameter_sweep_from_data(eta_list=[1.0], gamma_list=[0.0], seeds=[42], data_dir="data_K_class", fig_dir="figure_K_class", 
                                   alpha_0=1.0, N=2000, K=3, alpha_max=200, plot_idx=0, 
                                   alpha_min=1.0, d=10, ylim_D=None, plot_kwargs=None, plot_theory=True, start_idx=0):
    """
    Loads saved individual run arrays and dynamically rebuilds the sweep dictionary to reproduce envelope plots.
    """


    is_gamma_sweep = len(gamma_list) > 1
    sweep_vals = gamma_list if is_gamma_sweep else eta_list
    fixed_val = eta_list[0] if is_gamma_sweep else gamma_list[0]

    all_seeds_data = {}
    alpha_set = False
    
    for seed in seeds:
        for val in sweep_vals:
            eta_0 = fixed_val if is_gamma_sweep else val
            gamma = val if is_gamma_sweep else fixed_val

            filename = get_run_filename(N, K, eta_0, gamma, alpha_0, seed, alpha_max)
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"File {filepath} not found. Skipping seed {seed}, parameter {val}.")
                continue
                
            print(f"Loading data from: {filepath}")
            data = np.load(filepath)
            
            if not alpha_set:
                all_seeds_data["alpha"] = data["alpha"][start_idx:]
                alpha_set = True
                
            all_seeds_data[f"D_{val}_seed_{seed}"] = data["D"][start_idx:]
            all_seeds_data[f"Qeff_{val}_seed_{seed}"] = data["Qeff"][start_idx:]
            all_seeds_data[f"epsg_{val}_seed_{seed}"] = data["epsg"][start_idx:]
            all_seeds_data[f"celoss_{val}_seed_{seed}"] = data["celoss"][start_idx:]

    if not all_seeds_data:
        print("No valid data files found. Please run_parameter_sweep first.")
        return

    fig_base_name = generate_fig_base_name(N, K, eta_list, gamma_list, alpha_0, seeds, alpha_max, d, alpha_min)
    figs, axes = plot_results(all_seeds_data, is_gamma_sweep, sweep_vals, fixed_val, N, K, alpha_0, alpha_min, alpha_max, 
                 d, fig_base_name, plot_idx, plot_kwargs, fig_dir, ylim_D, seeds, plot_theory)

    return (figs, axes)

if __name__ == "__main__":
    # To use the new feature, you would pass it in via plot_kwargs like this:
    # my_kwargs = {
    #     'figsize': [3.2, 2.4], 
    #     'ax_dims': [0.2, 0.2, 0.75, 0.75],
    #     'figsize_small': [2.5, 2.0],
    #     'ax_dims_small': [0.25, 0.25, 0.70, 0.70]
    # }
    
    # Example 1: Multi-seed sweep over eta 
    run_parameter_sweep(
        eta_list=[0.2, 0.5, 0.8], 
        gamma_list=[0.0], 
        seeds=[42, 43, 44],
        num_test=100000,
        data_dir="data_K_class",
        fig_dir="figure_K_class",
        N=2000, 
        K=3, 
        alpha_max=200, 
        alpha_min=1.0, 
        spacing='log',
        first_step=1,
        num_saves=200,
        ylim_D=None,
        plot_theory=True
        # plot_kwargs=my_kwargs
    )