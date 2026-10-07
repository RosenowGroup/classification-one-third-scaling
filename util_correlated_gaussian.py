import numpy as np
import os
import itertools
from numba import njit
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure output directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("figures", exist_ok=True)

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
def run_sgd(T, J, eta, N_steps, test_X, eval_steps, var_xi, rng):
    """
    Runs online SGD with Numba for extreme performance.
    Evaluates test error and cross-entropy loss at specified eval_steps.
    """
    K = J.shape[0]
    N = J.shape[1]
    num_test = test_X.shape[0]
    num_saves = len(eval_steps)
    
    # Trackers
    alpha_hist = np.zeros(num_saves)
    epsg_hist = np.zeros(num_saves)
    loss_hist = np.zeros(num_saves)
    
    save_idx = 0
    
    for step in range(N_steps + 1):
        # --- Periodic Saving & Evaluation ---
        if save_idx < num_saves and step == eval_steps[save_idx]:
            alpha_hist[save_idx] = step / N
            
            errors = 0
            ce_loss = 0.0
            
            for i in range(num_test):
                xi_test = test_X[i]
                
                u = np.dot(T, xi_test) / np.sqrt(N)
                t = np.dot(J, xi_test) / np.sqrt(N)
                
                y = np.argmax(u)
                if y != np.argmax(t):
                    errors += 1
                
                # Cross-Entropy Loss
                max_t = np.max(t)
                exp_t = np.exp(t - max_t)
                p = exp_t / np.sum(exp_t)
                ce_loss -= np.log(p[y] + 1e-12) # Added small eps for numerical stability
                
            epsg_hist[save_idx] = errors / num_test
            loss_hist[save_idx] = ce_loss / num_test
            save_idx += 1
            
        if step == N_steps:
            break
            
        # --- Microscopic SGD Update ---
        # Sample input vector with specified variance
        xi = rng.normal(0.0, 1.0, N) * np.sqrt(var_xi)
        
        # Local fields (with 1/sqrt(N) normalization)
        u = np.dot(T, xi) / np.sqrt(N)
        t = np.dot(J, xi) / np.sqrt(N)
        
        y = np.argmax(u)
        
        # Softmax probabilities
        max_t = np.max(t)
        exp_t = np.exp(t - max_t)
        p = exp_t / np.sum(exp_t)
        
        # Cross-Entropy Gradient Step
        for k in range(K):
            delta_ky = 1.0 if k == y else 0.0
            J[k] = J[k] + (eta / np.sqrt(N)) * (delta_ky - p[k]) * xi
            
    return alpha_hist, epsg_hist, loss_hist

# ==========================================
# 3. Main Experiment Routine
# ==========================================

def run_experiment(N=2000, K=3, eta=0.5, alpha_max=200, seed=42, 
                   a=10.0, beta=0.0, gamma=0.0, 
                   num_test=10000, num_saves=200, 
                   spacing='log', first_step=1, plot_results=True, plot_kwargs=None):
    
    print(f"Running: N={N}, K={K}, eta={eta}, a={a}, beta={beta}, gamma={gamma}, seed={seed}")
    
    rng = np.random.default_rng(seed)
    N_steps = int(alpha_max * N)
    
    # Generate Variance arrays based on powerlaw
    indices = np.arange(N)
    var_xi = (a / (a + indices)) ** beta
    var_T = (a / (a + indices)) ** gamma
    
    # Initialize Teacher Vectors
    T_raw = rng.normal(0.0, 1.0, (K, N))
    T = np.zeros((K, N))
    for i in range(K):
        v = T_raw[i]
        for j in range(i):
            v -= np.dot(T[j], v) / N * T[j]
        v = v / np.sqrt(np.dot(v, v) / N)
        T[i] = v
    T = T * np.sqrt(var_T)
    
    # Initialize Student Vectors
    J = rng.normal(0.0, 1.0, (K, N))
    
    # Generate Test Data
    test_X_raw = rng.normal(0.0, 1.0, (num_test, N))
    test_X = test_X_raw * np.sqrt(var_xi)
    
    # Setup Evaluation Intervals
    if spacing == 'log':
        eval_steps = np.geomspace(first_step, N_steps, num_saves, dtype=int)
    else:
        eval_steps = np.linspace(first_step, N_steps, num_saves, dtype=int)
    eval_steps = np.unique(eval_steps)
    
    # Run Simulation
    alpha_hist, epsg_hist, loss_hist = run_sgd(T, J, eta, N_steps, test_X, eval_steps, var_xi, rng)
    
    # Construct Filename
    filename = f"results_N_{N}_K_{K}_eta_{eta}_seed_{seed}"
    if beta != 0.0 or gamma != 0.0:
        filename += f"_a_{a}"
    if beta != 0.0:
        filename += f"_beta_{beta}"
    if gamma != 0.0:
        filename += f"_gamma_{gamma}"
    filepath = os.path.join("data", filename + ".npz")
    
    # Save Data
    np.savez_compressed(filepath, alpha=alpha_hist, test_error=epsg_hist, test_loss=loss_hist)
    
    # Individual Plotting matching Reference Style
    if plot_results:
        if plot_kwargs is None:
            figsize = plt.rcParams['figure.figsize']
            ax_dims = [0.2, 0.2, 0.75, 0.75]
        else:
            figsize = plot_kwargs.get('figsize', plt.rcParams['figure.figsize'])
            ax_dims = plot_kwargs.get('ax_dims', [0.2, 0.2, 0.75, 0.75])

        # Figure Error
        fig_err = plt.figure(figsize=figsize)
        ax_err = fig_err.add_axes(ax_dims)
        ax_err.grid(False)
        ax_err.loglog(alpha_hist, epsg_hist, '-', color=colors[0], label='Test Error')
        
        # Power Law Guide (Theory proxy)
        alpha_mask = alpha_hist >= 1.0
        if np.any(alpha_mask):
            alpha_guide = alpha_hist[alpha_mask]
            theory_kwargs = dict(ls="--", dashes=[8, 4], linewidth=2.0, color='grey')
            
            err_max_first = max(epsg_hist[0], 1e-10)
            err_min_last = max(epsg_hist[-1], 1e-10)
            log_drop_err = np.abs(np.log10(err_max_first) - np.log10(err_min_last))
            err_guide_end = 10**(np.log10(err_min_last) - 0.1 * log_drop_err)
            C_err = err_guide_end * (alpha_guide[-1] ** (1/3))
            ax_err.loglog(alpha_guide, C_err * (alpha_guide ** (-1/3)), **theory_kwargs)

        theory_proxy = Line2D([0], [0], ls='--', dashes=[1.5, 2], linewidth=1.5, color='grey', label=r'$\propto \alpha^{-1/3}$')
        handles, labels = ax_err.get_legend_handles_labels()
        ax_err.legend(handles=handles + [theory_proxy], loc='best', handlelength=1.0)
        ax_err.set_xlabel(r"Training Time $\alpha$")
        ax_err.set_ylabel(r"Generalization Error $\epsilon_g$", labelpad=0)
        
        # Figure Loss
        fig_loss = plt.figure(figsize=figsize)
        ax_loss = fig_loss.add_axes(ax_dims)
        ax_loss.grid(False)
        ax_loss.loglog(alpha_hist, loss_hist, '-', color=colors[1], label='Test Loss (CE)')
        
        if np.any(alpha_mask):
            loss_max_first = max(loss_hist[0], 1e-10)
            loss_min_last = max(loss_hist[-1], 1e-10)
            log_drop_loss = np.abs(np.log10(loss_max_first) - np.log10(loss_min_last))
            loss_guide_end = 10**(np.log10(loss_min_last) - 0.1 * log_drop_loss)
            C_loss = loss_guide_end * (alpha_guide[-1] ** (1/3))
            ax_loss.loglog(alpha_guide, C_loss * (alpha_guide ** (-1/3)), **theory_kwargs)
            
        handles, labels = ax_loss.get_legend_handles_labels()
        ax_loss.legend(handles=handles + [theory_proxy], loc='best', handlelength=1.0)
        ax_loss.set_xlabel(r"Training Time $\alpha$")
        ax_loss.set_ylabel(r"Test Loss $\mathcal{L}_g$", labelpad=0)

        fig_err.savefig(os.path.join("figures", f"{filename}_err.pdf"))
        fig_loss.savefig(os.path.join("figures", f"{filename}_loss.pdf"))
        plt.show()


# ==========================================
# 4. Grid Search and Sweep Plotting Functions
# ==========================================

def grid_search(param_grid, fixed_params):
    """
    Executes a grid search over hyperparameters.
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for combo in combinations:
        kwargs = {**fixed_params, **combo}
        kwargs['plot_results'] = False 
        run_experiment(**kwargs)


def plot_sweep(sweep_param, param_values, fixed_params, seeds=[42], plot_kwargs=None):
    """
    Plots envelope results across multiple seeds for a sweep of a single parameter.
    Requires that the data has already been generated via run_experiment or grid_search.
    """
    if plot_kwargs is None:
        figsize = plt.rcParams['figure.figsize']
        ax_dims = [0.2, 0.2, 0.75, 0.75]
    else:
        figsize = plot_kwargs.get('figsize', plt.rcParams['figure.figsize'])
        ax_dims = plot_kwargs.get('ax_dims', [0.2, 0.2, 0.75, 0.75])

    fig_err = plt.figure(figsize=figsize)
    ax_err = fig_err.add_axes(ax_dims)
    ax_err.grid(False)

    fig_loss = plt.figure(figsize=figsize)
    ax_loss = fig_loss.add_axes(ax_dims)
    ax_loss.grid(False)

    all_errs_for_guide = []
    all_losses_for_guide = []
    alpha_base = None

    for idx, val in enumerate(param_values):
        color = colors[idx % len(colors)]
        if sweep_param == "beta":
            label_str = rf"$\beta={val}$"
        elif sweep_param == "eta":
            label_str = rf"$\eta={val}$"
        else:
            label_str = rf"{sweep_param}={val}"

        seed_errs = []
        seed_losses = []

        for seed in seeds:
            params = fixed_params.copy()
            params[sweep_param] = val
            params['seed'] = seed
            
            N, K, eta, s = params.get('N', 2000), params.get('K', 3), params.get('eta', 0.5), params.get('seed', 42)
            a, beta, gamma = params.get('a', 10.0), params.get('beta', 1.0), params.get('gamma', 1.0)
            
            filename = f"results_N_{N}_K_{K}_eta_{eta}_seed_{s}"
            if beta != 0.0 or gamma != 0.0:
                filename += f"_a_{a}"
            if beta != 0.0:
                filename += f"_beta_{beta}"
            if gamma != 0.0:
                filename += f"_gamma_{gamma}"
                
            filepath = os.path.join("data", filename + ".npz")
            
            if not os.path.exists(filepath):
                print(f"Warning: File {filepath} not found. Skipping seed {s}.")
                continue
                
            data = np.load(filepath)
            seed_errs.append(data['test_error'])
            seed_losses.append(data['test_loss'])
            if alpha_base is None:
                alpha_base = data['alpha']

        if len(seed_errs) == 0:
            continue
            
        seed_errs = np.array(seed_errs)
        seed_losses = np.array(seed_losses)
        
        all_errs_for_guide.extend(seed_errs)
        all_losses_for_guide.extend(seed_losses)

        # Plot Seed 0 as primary line
        ax_err.loglog(alpha_base, seed_errs[0], color=color, label=label_str)
        ax_loss.loglog(alpha_base, seed_losses[0], color=color, label=label_str)

        # Plot Envelopes
        if len(seeds) > 1:
            envelope_kwargs = dict(color=color, alpha=0.5, edgecolor="none")
            ax_err.fill_between(alpha_base, np.min(seed_errs, axis=0), np.max(seed_errs, axis=0), **envelope_kwargs)
            ax_loss.fill_between(alpha_base, np.min(seed_losses, axis=0), np.max(seed_losses, axis=0), **envelope_kwargs)

    # --- Power Law Guide (Theory Proxy) ---
    if alpha_base is not None and len(all_errs_for_guide) > 0:
        alpha_mask = alpha_base >= 1.0
        if np.any(alpha_mask):
            alpha_guide = alpha_base[alpha_mask]
            theory_kwargs = dict(ls="--", dashes=[2, 2.5], linewidth=2.0, color='grey')
            
            err_max_first = max(max([e[0] for e in all_errs_for_guide]), 1e-10)
            err_min_last = max(min([e[-1] for e in all_errs_for_guide]), 1e-10)
            log_drop_err = np.abs(np.log10(err_max_first) - np.log10(err_min_last))
            err_guide_end = 10**(np.log10(err_min_last) - 0.08 * log_drop_err)
            C_err = err_guide_end * (alpha_guide[-1] ** (1/3))
            ax_err.loglog(alpha_guide, C_err * (alpha_guide ** (-1/3)), **theory_kwargs)
            
            loss_max_first = max(max([l[0] for l in all_losses_for_guide]), 1e-10)
            loss_min_last = max(min([l[-1] for l in all_losses_for_guide]), 1e-10)
            log_drop_loss = np.abs(np.log10(loss_max_first) - np.log10(loss_min_last))
            loss_guide_end = 10**(np.log10(loss_min_last) - 0.08 * log_drop_loss)
            C_loss = loss_guide_end * (alpha_guide[-1] ** (1/3))
            ax_loss.loglog(alpha_guide, C_loss * (alpha_guide ** (-1/3)), **theory_kwargs)

    # --- Formatting Error Plot ---
    theory_proxy = Line2D([0], [0], ls='--', dashes=[1.5, 2], linewidth=1.5, color='grey', label=r'$\propto \alpha^{-1/3}$')
    
    handles, labels = ax_err.get_legend_handles_labels()
    ax_err.legend(handles=handles + [theory_proxy], loc='best', handlelength=1.0,
                   labelspacing=0.1, frameon=False, borderpad=0.1, borderaxespad=0.7,
                   handletextpad=0.5)
    ax_err.set_xlabel(r"Training Time $\alpha$")
    ax_err.set_ylabel(r"Generalization Error $\epsilon_g$", labelpad=0)
    
    # --- Formatting Loss Plot ---
    handles, labels = ax_loss.get_legend_handles_labels()
    ax_loss.legend(handles=handles + [theory_proxy], loc='best', handlelength=1.0,
                   labelspacing=0.1, frameon=False, borderpad=0.1, borderaxespad=0.7,
                   handletextpad=0.5)
    ax_loss.set_xlabel(r"Training Time $\alpha$")
    ax_loss.set_ylabel(r"Test Loss $\mathcal{L}_g$", labelpad=0)
    
    # --- Constructing save filename ---
    save_params = fixed_params.copy()
    if len(param_values) > 0:
        save_params[sweep_param] = param_values[0] 
        
    N_s, K_s, eta_s = save_params.get('N', 2000), save_params.get('K', 3), save_params.get('eta', 0.5)
    a_s, beta_s, gamma_s = save_params.get('a', 10.0), save_params.get('beta', 1.0), save_params.get('gamma', 1.0)
    seed_str = "_".join([str(s) for s in seeds])
    
    save_filename = f"sweep_{sweep_param}_N_{N_s}_K_{K_s}_eta_{eta_s}_seeds_{seed_str}"
    if beta_s != 0.0 or gamma_s != 0.0:
        save_filename += f"_a_{a_s}"
    if beta_s != 0.0:
        save_filename += f"_beta_{beta_s}"
    if gamma_s != 0.0:
        save_filename += f"_gamma_{gamma_s}"
        
    fig_err.savefig(os.path.join("figures", save_filename + "_err.pdf"))
    fig_loss.savefig(os.path.join("figures", save_filename + "_loss.pdf"))
    print("Finished successfully. Figures saved.")
    plt.show()

if __name__ == "__main__":
    my_kwargs = {
        'figsize': [3.2, 2.4], 
        'ax_dims': [0.2, 0.2, 0.75, 0.75]
    }

    # 1. Example Grid Search Execution (Adding multi-seed into the grid)
    fixed_params = {'N': 2000, 'K': 3, 'alpha_max': 200, 'a': 10.0, 'gamma': 0.0}
    param_grid = {'eta': [0.1, 0.5, 1.0], 'beta': [1.0, 1.5], 'seed': [42, 43, 44]}
    
    print("\nStarting Grid Search...")
    grid_search(param_grid, fixed_params)
    
    # 2. Plotting the Multi-Seed Sweep across eta
    print("\nPlotting Parameter Sweep...")
    sweep_fixed = {'N': 2000, 'K': 3, 'a': 10.0, 'beta': 1.5, 'gamma': 0.0}
    plot_sweep(
        sweep_param='eta', 
        param_values=[0.1, 0.5, 1.0], 
        seeds=[42, 43, 44], 
        fixed_params=sweep_fixed,
        plot_kwargs=my_kwargs
    )