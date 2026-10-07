"""Boundary-layer gap diagnostics for the isotropic softmax model."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numba import njit


_HERE = Path(__file__).resolve().parent
_CACHE_VERSION = 3
_COLORS = ["#0C5DA5", "#00B945", "#FF9500"]

try:
    plt.style.use(str(_HERE / "plt_style.pstyle"))
except OSError:
    pass

plt.rcParams.update(
    {
        "font.size": 7.8,
        "figure.titlesize": 9,
        "legend.fontsize": 7,
        "axes.titlesize": 9,
        "axes.labelsize": 8.8,
        "figure.figsize": [2.45, 1.5],
        "mathtext.fontset": "cm",
    }
)


def _validate(
    N,
    K,
    eta,
    alphas,
    density_alphas,
    seeds,
    n_eval,
    eval_batch,
    n_bins,
    gap_window,
    error_x_max,
    gradient_x_max,
):
    if K < 2 or N < K:
        raise ValueError("Require N >= K >= 2.")
    if eta <= 0 or n_eval <= 0 or eval_batch <= 0:
        raise ValueError("eta, n_eval, and eval_batch must be positive.")
    if (
        n_bins < 2
        or gap_window <= 0
        or not np.isfinite(error_x_max)
        or error_x_max <= 0
        or not np.isfinite(gradient_x_max)
        or gradient_x_max <= 0
    ):
        raise ValueError("n_bins must be at least 2 and all gap ranges must be positive.")
    if not alphas or any(not np.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("alphas must be nonempty, finite, and positive.")
    if tuple(sorted(set(alphas))) != tuple(alphas):
        raise ValueError("alphas must be sorted and unique.")
    if len(set(density_alphas)) != 3 or any(a not in alphas for a in density_alphas):
        raise ValueError("density_alphas must contain exactly three distinct values from alphas.")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds must be nonempty, unique, and nonnegative.")
    for alpha in alphas:
        steps = alpha * N
        if not np.isclose(steps, round(steps), atol=1e-10, rtol=0.0):
            raise ValueError(f"alpha={alpha:g} does not correspond to an integer SGD step.")


def _make_teacher_and_student(N: int, K: int, seed: int):
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(K, N))
    teacher = np.zeros_like(raw)
    for a in range(K):
        vector = raw[a].copy()
        for b in range(a):
            vector -= (teacher[b] @ vector / N) * teacher[b]
        teacher[a] = vector / np.sqrt(vector @ vector / N)
    student = rng.normal(size=(K, N))
    online_seed = int(rng.integers(1, 2_000_000_000))
    return teacher, student, online_seed


@njit(cache=True)
def _train_to_checkpoints(teacher, student_initial, eta, checkpoint_steps, online_seed):
    np.random.seed(online_seed)
    K, N = student_initial.shape
    student = student_initial.copy()
    snapshots = np.empty((checkpoint_steps.size, K, N), dtype=np.float64)
    xi = np.empty(N)
    u = np.empty(K)
    h = np.empty(K)
    p = np.empty(K)
    save_index = 0
    max_step = int(checkpoint_steps[-1])
    sqrt_N = math.sqrt(N)

    for step in range(max_step + 1):
        if save_index < checkpoint_steps.size and step == checkpoint_steps[save_index]:
            snapshots[save_index] = student
            save_index += 1
        if step == max_step:
            break

        for i in range(N):
            xi[i] = np.random.normal()
        for a in range(K):
            teacher_sum = 0.0
            student_sum = 0.0
            for i in range(N):
                teacher_sum += teacher[a, i] * xi[i]
                student_sum += student[a, i] * xi[i]
            u[a] = teacher_sum / sqrt_N
            h[a] = student_sum / sqrt_N

        y = np.argmax(u)
        h_max = np.max(h)
        normalizer = 0.0
        for a in range(K):
            p[a] = math.exp(h[a] - h_max)
            normalizer += p[a]
        scale = eta / sqrt_N
        for a in range(K):
            p[a] /= normalizer
            coefficient = ((1.0 if a == y else 0.0) - p[a]) * scale
            for i in range(N):
                student[a, i] += coefficient * xi[i]

    return snapshots


def _order_parameters(student: np.ndarray, teacher: np.ndarray):
    K, N = student.shape
    scale = math.sqrt(K / (K - 1.0))
    teacher_hat = scale * (teacher[0] - teacher.mean(axis=0))
    student_hat = scale * (student[0] - student.mean(axis=0))
    D = float(student_hat @ teacher_hat / N)
    Qeff = float(student_hat @ student_hat / N)
    residual = student_hat - D * teacher_hat
    Delta = float(residual @ residual / N)
    if not np.isclose(Delta, Qeff - D * D, atol=2e-10, rtol=2e-10):
        raise RuntimeError("The geometric and algebraic definitions of Delta disagree.")
    return D, Delta


def _joint_field_factor(student: np.ndarray, teacher: np.ndarray):
    N = student.shape[1]
    Cuu = teacher @ teacher.T / N
    Cuh = teacher @ student.T / N
    Chh = student @ student.T / N
    covariance = np.block([[Cuu, Cuh], [Cuh.T, Chh]])
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 1e-11 * max(1.0, float(np.max(eigenvalues)))
    if float(np.min(eigenvalues)) < -tolerance:
        raise RuntimeError("The joint field covariance is not positive semidefinite.")
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))[None, :]


def _measure_histograms(student, teacher, median_edges, error_edges, gradient_edges, n_eval, eval_batch, seed):
    K = student.shape[0]
    factor = _joint_field_factor(student, teacher)
    rng = np.random.default_rng(seed)
    shape = median_edges.size - 1
    result = {
        "error_count": np.zeros(shape, dtype=np.int64),
        "error_sum": np.zeros(shape, dtype=np.int64),
        "gradient_count": np.zeros(shape, dtype=np.int64),
        "gradient_sum": np.zeros(shape),
        "median_error": np.zeros(shape, dtype=np.int64),
        "median_gradient": np.zeros(shape),
        "total_error": 0,
        "total_gradient": 0.0,
    }

    done = 0
    while done < n_eval:
        size = min(eval_batch, n_eval - done)
        fields = rng.standard_normal((size, 2 * K)) @ factor.T
        u, h = fields[:, :K], fields[:, K:]
        rows = np.arange(size)
        y, prediction = np.argmax(u, axis=1), np.argmax(h, axis=1)

        top = np.partition(u, K - 2, axis=1)[:, -2:]
        gap = top[:, 1] - top[:, 0]

        shifted = h - np.max(h, axis=1, keepdims=True)
        exp_h = np.exp(shifted)
        probabilities = exp_h / np.sum(exp_h, axis=1, keepdims=True)
        error = prediction != y
        gradient = np.maximum(u[rows, y] - np.sum(probabilities * u, axis=1), 0.0)

        result["error_count"] += np.histogram(gap, bins=error_edges)[0]
        result["error_sum"] += np.histogram(gap[error], bins=error_edges)[0]
        result["gradient_count"] += np.histogram(gap, bins=gradient_edges)[0]
        result["gradient_sum"] += np.histogram(gap, bins=gradient_edges, weights=gradient)[0]
        result["median_error"] += np.histogram(gap[error], bins=median_edges)[0]
        result["median_gradient"] += np.histogram(gap, bins=median_edges, weights=gradient)[0]

        result["total_error"] += int(np.sum(error))
        result["total_gradient"] += float(np.sum(gradient))
        done += size

    return result


def _cache_key(
    N,
    K,
    eta,
    alphas,
    density_alphas,
    seeds,
    n_eval,
    eval_batch,
    n_bins,
    gap_window,
    error_x_max,
    gradient_x_max,
):
    settings = (
        _CACHE_VERSION,
        N,
        K,
        eta,
        alphas,
        density_alphas,
        seeds,
        n_eval,
        eval_batch,
        n_bins,
        gap_window,
        error_x_max,
        gradient_x_max,
    )
    digest = hashlib.sha256(repr(settings).encode("utf-8")).hexdigest()[:12]
    return f"v{_CACHE_VERSION}_N{N}_K{K}_eta{eta:g}_neval{n_eval}_{digest}"


def _load_cache(path: Path, key: str, n_alpha: int, n_bins: int):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["cache_key"]) != key:
                return None
            result = {name: data[name].copy() for name in data.files if name != "cache_key"}
    except (OSError, ValueError, KeyError):
        return None

    vector_keys = ("D", "Delta", "total_error", "total_gradient")
    histogram_keys = (
        "error_count",
        "error_sum",
        "gradient_count",
        "gradient_sum",
        "median_error",
        "median_gradient",
    )
    edge_keys = ("median_edges", "error_edges", "gradient_edges")
    required = set(vector_keys + histogram_keys + edge_keys)
    if not required.issubset(result):
        return None
    if any(result[name].shape != (n_alpha,) for name in vector_keys):
        return None
    if any(result[name].shape != (n_alpha, n_bins) for name in histogram_keys):
        return None
    if result["median_edges"].shape != (n_alpha, n_bins + 1):
        return None
    if result["error_edges"].shape != (n_bins + 1,):
        return None
    if result["gradient_edges"].shape != (n_bins + 1,):
        return None
    if not np.all(np.isfinite(result["D"])) or not np.all(np.isfinite(result["Delta"])):
        return None
    return result
    return result


def _save_cache(path: Path, key: str, result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cache_key=np.asarray(key), **result)


def _weighted_histogram_median(histogram, total, edges):
    if total <= 0:
        raise RuntimeError("No contributing samples were observed; increase n_eval.")
    cumulative = np.cumsum(histogram)
    target = 0.5 * total
    if cumulative[-1] < target:
        raise RuntimeError("The gap window contains less than half the weight; increase gap_window.")
    index = int(np.searchsorted(cumulative, target, side="left"))
    below = 0.0 if index == 0 else float(cumulative[index - 1])
    fraction = (target - below) / float(histogram[index])
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def _mean_finite(values: np.ndarray):
    count = np.sum(np.isfinite(values), axis=0)
    return np.divide(
        np.nansum(values, axis=0),
        count,
        out=np.full(values.shape[1:], np.nan),
        where=count > 0,
    )


def _aggregate(records, alphas, seeds):
    stacked = {name: np.stack([records[seed][name] for seed in seeds]) for name in records[seeds[0]]}
    results = {
        "alpha": np.asarray(alphas),
        "seeds": np.asarray(seeds),
        "D_per_seed": stacked["D"],
        "Delta_per_seed": stacked["Delta"],
        "D": np.mean(stacked["D"], axis=0),
        "Delta": np.mean(stacked["Delta"], axis=0),
    }
    conditional, medians = {}, {}
    median_edges = stacked["median_edges"][0]
    for quantity in ("error", "gradient"):
        counts = stacked[f"{quantity}_count"]
        conditional[quantity] = np.divide(
            stacked[f"{quantity}_sum"],
            counts,
            out=np.full(counts.shape, np.nan, dtype=float),
            where=counts > 0,
        )
        medians[quantity] = np.empty((len(seeds), len(alphas)))
        for seed_index in range(len(seeds)):
            for alpha_index in range(len(alphas)):
                medians[quantity][seed_index, alpha_index] = _weighted_histogram_median(
                    stacked[f"median_{quantity}"][seed_index, alpha_index],
                    stacked[f"total_{quantity}"][seed_index, alpha_index],
                    median_edges[alpha_index],
                )

    results["teacher"] = {
        "error_gap": 0.5 * (stacked["error_edges"][0, 1:] + stacked["error_edges"][0, :-1]),
        "gradient_gap": 0.5
        * (stacked["gradient_edges"][0, 1:] + stacked["gradient_edges"][0, :-1]),
        "error_per_seed": conditional["error"],
        "gradient_per_seed": conditional["gradient"],
        "error": _mean_finite(conditional["error"]),
        "gradient": _mean_finite(conditional["gradient"]),
        "error_median_per_seed": medians["error"],
        "gradient_median_per_seed": medians["gradient"],
        "error_median": np.mean(medians["error"], axis=0),
        "gradient_median": np.mean(medians["gradient"], axis=0),
    }
    return results


def _new_figure():
    figure = plt.figure(figsize=[2.45, 1.5])
    axis = figure.add_axes([0.24, 0.31, 0.72, 0.62])
    axis.grid(False)
    return figure, axis


def _alpha_text(alpha: float):
    exponent = int(math.floor(math.log10(alpha)))
    coefficient = alpha / 10**exponent
    if np.isclose(coefficient, 1.0):
        return rf"10^{{{exponent}}}"
    #if np.isclose(coefficient, round(coefficient)):
    coefficient = int(round(coefficient))
    return rf"{coefficient:g}\times 10^{{{exponent}}}"


def _legend(axis):
    axis.legend(
        loc="best",
        handlelength=1.0,
        labelspacing=0.1,
        frameon=False,
        borderpad=0.1,
        borderaxespad=0.7,
        handletextpad=0.5,
    )


def _conditional_plot(results, quantity, density_alphas, x_max):
    figure, axis = _new_figure()
    alphas = results["alpha"]
    for color, alpha in zip(_COLORS, density_alphas, strict=True):
        index = int(np.flatnonzero(alphas == alpha)[0])
        axis.plot(
            results["teacher"][f"{quantity}_gap"],
            results["teacher"][quantity][index],
            color=color,
            label=rf"$\alpha={_alpha_text(alpha)}\ (D={results['D'][index]:.3g})$",
        )

    axis.set_xlim(-0.05 * x_max, 1.05 * x_max)
    axis.set_xlabel(r"teacher gap $G = u_y - \max_{a\neq y}u_a$")
    if quantity == "error":
        axis.set_ylabel(r"$\left.\epsilon_g\right|_G$", labelpad=0)
    else:
        axis.set_ylabel(
            r"$\left\langle\left.-\frac{\partial \ell}{\partial D}\right|_G\right\rangle$",
            labelpad=0,
        )
    axis.yaxis.label.set_fontsize(8)
    _legend(axis)
    return figure, axis


def _median_plot(results, quantity):
    figure, axis = _new_figure()
    D = results["D"]
    median = results["teacher"][f"{quantity}_median"]
    if np.any(D <= 0) or np.any(median <= 0):
        raise RuntimeError("Median scaling plots require positive D and positive medians.")
    order = np.argsort(D)
    D, median = D[order], median[order]
    axis.loglog(D, median, "o-", markersize=3, color=_COLORS[0], label="numerics")
    axis.loglog(D, median[-1] * D[-1] / D, "--", dashes=[2, 2.5], linewidth=2, color="grey", label=r"$1/D$")

    axis.set_xlabel(r"alignment $D$")
    if quantity == "error":
        axis.set_ylabel(r"$G_{\epsilon_g}^{\rm median}(D)$", labelpad=0)
    else:
        axis.set_ylabel(
            r"$G_{-\frac{\partial \ell}{\partial D}}^{\rm median}(D)$",
            labelpad=0,
        )
    axis.yaxis.label.set_fontsize(8)
    _legend(axis)
    return figure, axis


def _plot_results(results, density_alphas, fig_dir: Path, error_x_max, gradient_x_max):
    figures, axes = {}, {}
    for quantity, x_max in (("error", error_x_max), ("gradient", gradient_x_max)):
        conditional_key = f"teacher_{quantity}_conditional"
        figures[conditional_key], axes[conditional_key] = _conditional_plot(
            results, quantity, density_alphas, x_max
        )

        median_key = f"teacher_{quantity}_median"
        figures[median_key], axes[median_key] = _median_plot(results, quantity)

    fig_dir.mkdir(parents=True, exist_ok=True)
    for key, figure in figures.items():
        figure.savefig(fig_dir / f"mechanism_{key}.pdf")
    plt.show()
    return figures, axes


def run_mechanism(
    N=200,
    K=3,
    eta=0.5,
    alphas=(100, 300, 1000, 3000, 10000, 30000, 100000),
    density_alphas=(100, 3000, 100000),
    seeds=(0, 1, 2, 3, 4, 5),
    n_eval=2_000_000,
    eval_batch=200_000,
    n_bins=400,
    gap_window=10,
    error_x_max=0.6,
    gradient_x_max=2.7,
    data_dir=None,
    fig_dir=None,
    force_rerun=False,
):
    """Run the mechanism experiment and return ``(results, figures, axes)``."""

    alphas = tuple(float(alpha) for alpha in alphas)
    density_alphas = tuple(float(alpha) for alpha in density_alphas)
    seeds = tuple(int(seed) for seed in seeds)
    _validate(
        N,
        K,
        eta,
        alphas,
        density_alphas,
        seeds,
        n_eval,
        eval_batch,
        n_bins,
        gap_window,
        error_x_max,
        gradient_x_max,
    )

    checkpoint_steps = np.asarray([round(alpha * N) for alpha in alphas], dtype=np.int64)
    data_dir = _HERE / "data" / "mechanism" if data_dir is None else Path(data_dir)
    fig_dir = _HERE / "figures" if fig_dir is None else Path(fig_dir)
    key = _cache_key(
        N,
        K,
        eta,
        alphas,
        density_alphas,
        seeds,
        n_eval,
        eval_batch,
        n_bins,
        gap_window,
        error_x_max,
        gradient_x_max,
    )
    cache_paths = {seed: data_dir / f"mechanism_{key}_seed{seed}.npz" for seed in seeds}

    records, pending = {}, {}
    for seed in seeds:
        cached = None if force_rerun else _load_cache(cache_paths[seed], key, len(alphas), n_bins)
        if cached is not None:
            print(f"Loading mechanism cache for seed {seed}.")
            records[seed] = cached
            continue
        print(f"Training mechanism checkpoints for seed {seed}.")
        teacher, student_initial, online_seed = _make_teacher_and_student(N, K, seed)
        snapshots = _train_to_checkpoints(teacher, student_initial, eta, checkpoint_steps, online_seed)
        D, Delta = zip(*[_order_parameters(student, teacher) for student in snapshots], strict=True)
        pending[seed] = {
            "teacher": teacher,
            "snapshots": snapshots,
            "D": np.asarray(D),
            "Delta": np.asarray(Delta),
        }

    all_D = np.stack([records[s]["D"] if s in records else pending[s]["D"] for s in seeds])
    all_Delta = np.stack([records[s]["Delta"] if s in records else pending[s]["Delta"] for s in seeds])
    D_min = np.min(all_D, axis=0)
    if np.any(D_min <= 0):
        raise RuntimeError("All checkpoint alignments D must be positive.")
    gap_scale = np.maximum(1.0, np.sqrt(np.max(all_Delta, axis=0)))
    unit_edges = np.linspace(0.0, 1.0, n_bins + 1)
    median_edges = gap_window * gap_scale[:, None] / D_min[:, None] * unit_edges
    error_edges = np.linspace(0.0, error_x_max, n_bins + 1)
    gradient_edges = np.linspace(0.0, gradient_x_max, n_bins + 1)

    for seed, cached in records.items():
        if not (
            np.allclose(cached["median_edges"], median_edges)
            and np.allclose(cached["error_edges"], error_edges)
            and np.allclose(cached["gradient_edges"], gradient_edges)
        ):
            raise RuntimeError(f"Cached bin edges for seed {seed} are inconsistent; use force_rerun=True.")

    for seed, run in pending.items():
        print(f"Evaluating fresh Gaussian fields for seed {seed}.")
        record = {
            "D": run["D"],
            "Delta": run["Delta"],
            "median_edges": median_edges,
            "error_edges": error_edges,
            "gradient_edges": gradient_edges,
        }
        histogram_names = (
            "error_count",
            "error_sum",
            "gradient_count",
            "gradient_sum",
            "median_error",
            "median_gradient",
        )
        for name in histogram_names:
            dtype = np.int64 if name in {"error_count", "error_sum", "gradient_count", "median_error"} else float
            record[name] = np.zeros((len(alphas), n_bins), dtype=dtype)
        record["total_error"] = np.zeros(len(alphas), dtype=np.int64)
        record["total_gradient"] = np.zeros(len(alphas))

        for index, student in enumerate(run["snapshots"]):
            measured = _measure_histograms(
                student,
                run["teacher"],
                median_edges[index],
                error_edges,
                gradient_edges,
                n_eval,
                eval_batch,
                np.random.SeedSequence([seed, int(checkpoint_steps[index]), 271828]),
            )
            for name in histogram_names:
                record[name][index] = measured[name]
            record["total_error"][index] = measured["total_error"]
            record["total_gradient"][index] = measured["total_gradient"]
        _save_cache(cache_paths[seed], key, record)
        records[seed] = record

    results = _aggregate(records, alphas, seeds)
    figures, axes = _plot_results(results, density_alphas, fig_dir, error_x_max, gradient_x_max)
    return results, figures, axes


__all__ = ["run_mechanism"]
