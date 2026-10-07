from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# 1. Target Style Configurations
figsize = [2.55, 1.7]
ax_dims = [0.17, 0.2, 0.80, 0.78]

plt.style.use('plt_style.pstyle')
plt.rcParams["font.size"] = 7.8        # tick label
plt.rcParams["figure.titlesize"] = 9   # title
plt.rcParams["legend.fontsize"] = 7    # legend
plt.rcParams["axes.titlesize"] = 9     # axes title
plt.rcParams["axes.labelsize"] = 8.8   # axes label
plt.rcParams['figure.figsize'] = figsize
plt.rcParams["mathtext.fontset"] = "cm"

colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#8c564b', '#bcbd22', '#adfc03']

# 2. Experiment Parameters
lr_list = [0.0003]
seed_list = [0, 1, 2, 3]
num_examples = 1800000
test_split = 0.01
bs = 1
use_whitening = True
figure_dir = Path("figures")
figure_dir.mkdir(parents=True, exist_ok=True)

def data_title_gen(lr, seed, true_labels, num_examples, bs, use_whitening=True):
    title = f"lr={lr:.2e}_seed={seed}_examples={num_examples}_bs={bs}"
    if not use_whitening:
        title += "_unwhitened"
    if true_labels:
        title += "_TrueLabels"
    else:
        title += "_TeacherLabels"
    return title

# 3. Main Plotting Loop
for true_labels in [True, False]:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(ax_dims)
    ax.grid(False) 

    max_test_acc = 0
    loaded_steps = None
    x_log_spaced = None
    
    for idx, lr in enumerate(lr_list):
        test_acc_list = []
        
        for seed in seed_list:
            loaded = np.load("data/cifar5m_vit_data_" + data_title_gen(lr, seed, true_labels, num_examples, bs, use_whitening) + ".npz", allow_pickle=True)
            
            if loaded_steps is None:
                loaded_steps = loaded["steps"]
                
            test_acc_list.append(loaded["test_acc"])
            max_test_acc = np.max((max_test_acc, loaded["test_acc"][-1]))
        
        # Calculate raw error arrays
        test_acc_list = np.array(test_acc_list)
        errors = 1 - test_acc_list
        
        # --- DATA ADJUSTMENT: Log-spaced interpolation ---
        # Filter out 0 to safely use geomspace and loglog
        valid_idx = loaded_steps > 0
        x_raw = loaded_steps[valid_idx]
        
        # Create num_points log-spaced points across the domain
        num_points = 150
        if x_log_spaced is None:
            x_log_spaced = np.geomspace(x_raw[0], x_raw[-1], num_points)
            
        # Interpolate errors for all seeds onto the new log-spaced x-axis
        interp_errors = np.zeros((len(seed_list), num_points))
        for s_idx in range(len(seed_list)):
            interp_errors[s_idx] = np.interp(x_log_spaced, x_raw, errors[s_idx, valid_idx])
            
        mean_errors = np.mean(interp_errors, axis=0)
        min_errors = np.min(interp_errors, axis=0)
        max_errors = np.max(interp_errors, axis=0)

        color = colors[idx % len(colors)]
        label_str = rf"$\eta$={lr:.1e}"
        
        # Plot Mean and Envelope using the smoothed data
        ax.loglog(x_log_spaced, mean_errors, color=color, label=label_str)
        ax.fill_between(x_log_spaced, min_errors, max_errors, color=color, alpha=0.5, edgecolor="none")

    # Power Law Guide (Theory proxy) evaluated on the smoothed domain
    theory_kwargs = dict(ls="--", dashes=[2, 2.5], linewidth=2.0, color='grey')
    if true_labels:
        idx_low = 80
        powerlaw_line = 2.0*(x_log_spaced[idx_low:] / x_log_spaced[-1])**-(1/3) * (1 - max_test_acc)
    else:
        idx_low = 60
        powerlaw_line = 0.5*(x_log_spaced[idx_low:] / x_log_spaced[-1])**-(1/3) * (1 - max_test_acc)
    ax.loglog(x_log_spaced[idx_low:], powerlaw_line, **theory_kwargs)

    # Exact labels from target style
    ax.set_xlabel(r"Training Time $\alpha$")
    ax.set_ylabel(r"Generalization Error $\epsilon_g$", labelpad=0) 

    # Legend Construction (Frameless, Custom proxy handles)
    theory_proxy = Line2D([0], [0], ls='--', dashes=[1.5, 2], linewidth=1.5, color='grey', label=r'$\propto \alpha^{-1/3}$')
    handles, labels = ax.get_legend_handles_labels()
    
    ax.legend(handles=handles + [theory_proxy], loc='best', handlelength=1.0,
              labelspacing=0.1, frameon=False, borderpad=0.1, borderaxespad=0.7,
              handletextpad=0.5)

    #Save output (optional)
    label_type = "TrueLabels" if true_labels else "TeacherLabels"
    fig.savefig(figure_dir / f"cifar5m_{label_type}_dynamics.pdf")
