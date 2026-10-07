"""Fit the Pythia top-1 errors and create the two manuscript figures."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from scipy import optimize
import torch


_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "pythia"
_STYLE_PATH = _HERE / "plt_style.pstyle"

_STEPS = np.array(
    [128, 256, 512, 1000, 2000, 3000, 6000, 12000, 16000, 32000, 48000, 79000, 143000]
)
_MODELS = (
    ("410M", "410m", 3e-4),
    ("2.8B", "2p8b", 1.6e-4),
    ("12B", "12b", 1.2e-4),
)
_NOMINAL_WINDOW = (2000, 79000)
_ALTERNATIVE_WINDOWS = (
    (2000, 143000),
    (2000, 48000),
    (3000, 79000),
    (3000, 143000),
    (3000, 48000),
    (1000, 79000),
    (1000, 143000),
    (1000, 48000),
)

_COLORS = ("#0C5DA5", "#00B945", "#FF9500")
_DARK_COLORS = ("#003A6D", "#007029", "#C57300")
_FIT_LINE_STYLE = {"ls": "--", "dashes": [2, 1.8], "linewidth": 2.0}


def _configure_plot_style() -> None:
    if not _STYLE_PATH.is_file():
        raise FileNotFoundError(f"Could not find Matplotlib style: {_STYLE_PATH}")
    plt.style.use(str(_STYLE_PATH))
    plt.rcParams.update(
        {
            "font.size": 7.8,
            "figure.titlesize": 9,
            "legend.fontsize": 7.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.8,
            "figure.figsize": [3.2, 2.4],
            "mathtext.fontset": "cm",
        }
    )


def _load_top1_errors() -> np.ndarray:
    top1_errors = []
    for model, slug, _ in _MODELS:
        path = _DATA_DIR / f"pythia-logit-{slug}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path.name}; run or copy the {model} evaluation first")

        result = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(result, dict):
            raise ValueError(f"{path.name} must contain a dictionary")
        missing = {"steps", "completed_steps", "top1_error"}.difference(result)
        if missing:
            raise ValueError(f"{path.name} is missing {sorted(missing)}")

        saved_steps = torch.as_tensor(result["steps"]).cpu().numpy().astype(int)
        completed_steps = (
            torch.as_tensor(result["completed_steps"]).cpu().numpy().astype(int)
        )
        if not np.array_equal(saved_steps, _STEPS) or not np.array_equal(
            completed_steps, _STEPS
        ):
            raise ValueError(
                f"{path.name} is incomplete or uses a different checkpoint schedule"
            )

        values = torch.as_tensor(result["top1_error"]).cpu().numpy()
        if (
            values.shape != _STEPS.shape
            or not np.all(np.isfinite(values))
            or np.any(values <= 0)
        ):
            raise ValueError(
                f"{path.name} top1_error must contain {_STEPS.size} positive finite values"
            )
        top1_errors.append(values)
    return np.stack(top1_errors)


def _residual_aware_times() -> np.ndarray:
    total_steps = 143000
    warmup_steps = total_steps * 0.01
    all_steps = np.arange(1, total_steps + 1, dtype=float)
    progress = (all_steps - warmup_steps) / (total_steps * 0.99)
    learning_rate_multiplier = np.where(
        all_steps < warmup_steps,
        all_steps / warmup_steps,
        0.5 * (1 + np.cos(np.pi * progress)) * 0.9 + 0.1,
    )

    peak_learning_rates = np.array([model[2] for model in _MODELS])
    h_multiplier = np.cumsum(learning_rate_multiplier)[_STEPS - 1]
    multiplier_at_checkpoints = learning_rate_multiplier[_STEPS - 1]
    h = peak_learning_rates[:, None] * h_multiplier[None, :]
    eta = peak_learning_rates[:, None] * multiplier_at_checkpoints[None, :]
    return h / eta ** (3 / 2)


def _window_mask(window: tuple[int, int]) -> np.ndarray:
    start_step, end_step = window
    mask = (_STEPS >= start_step) & (_STEPS <= end_step)
    if mask.sum() < 4:
        raise ValueError(f"Fit window {window} contains fewer than four checkpoints")
    return mask


def _log_power_law(
    x: np.ndarray, coefficient: float, exponent: float, floor: float
) -> np.ndarray:
    return np.log(coefficient / x**exponent + floor)


def _fit_power_law(
    x: np.ndarray, y: np.ndarray, window: tuple[int, int]
) -> dict[str, float]:
    mask = _window_mask(window)
    floor_cap = y[mask][-1]
    parameters, covariance = optimize.curve_fit(
        _log_power_law,
        x[mask],
        np.log(y[mask]),
        maxfev=100000,
        p0=[1, 0.33, floor_cap],
        bounds=([0, 0, 0], [np.inf, np.inf, floor_cap]),
    )
    standard_errors = np.sqrt(np.diag(covariance))
    return {
        "coefficient": parameters[0],
        "exponent": parameters[1],
        "floor": parameters[2],
        "coefficient_se": standard_errors[0],
        "exponent_se": standard_errors[1],
        "floor_se": standard_errors[2],
    }


def _fit_models(
    x_values: np.ndarray, top1_errors: np.ndarray
) -> list[dict[str, float]]:
    fits = []
    windows = (_NOMINAL_WINDOW, *_ALTERNATIVE_WINDOWS)
    for x, y in zip(x_values, top1_errors):
        candidates = [_fit_power_law(x, y, window) for window in windows]
        nominal = candidates[0]
        lower_bound = min(
            fit["exponent"] - fit["exponent_se"] for fit in candidates
        )
        upper_bound = max(
            fit["exponent"] + fit["exponent_se"] for fit in candidates
        )
        nominal["exponent_error_minus"] = nominal["exponent"] - lower_bound
        nominal["exponent_error_plus"] = upper_bound - nominal["exponent"]
        fits.append(nominal)
    return fits


def _legend_exponent(fit: dict[str, float]) -> str:
    return (
        rf"{fit['exponent']:.3f}"
        rf"^{{+{fit['exponent_error_plus']:.3f}}}"
        rf"_{{-{fit['exponent_error_minus']:.3f}}}"
    )


def _add_legend(axes) -> None:
    fit_proxy = Line2D(
        [0],
        [0],
        ls="--",
        dashes=[1.5, 2],
        linewidth=1.5,
        color="grey",
        label="fit",
    )
    handles, labels = axes.get_legend_handles_labels()
    axes.legend(
        handles=handles + [fit_proxy],
        labels=labels + [fit_proxy.get_label()],
        loc="upper right",
        handlelength=1.0,
        labelspacing=0.1,
        frameon=False,
        borderpad=0.1,
        borderaxespad=0.4,
        handletextpad=0.5,
    )


def _raw_plot(
    x_values: np.ndarray,
    top1_errors: np.ndarray,
    fits: list[dict[str, float]],
) -> Figure:
    figure = plt.figure(figsize=[3.2, 2.4])
    axes = figure.add_axes([0.2, 0.2, 0.75, 0.75])
    axes.grid(False)
    fit_mask = _window_mask(_NOMINAL_WINDOW)

    for index, ((model, _, _), x, y, fit) in enumerate(
        zip(_MODELS, x_values, top1_errors, fits)
    ):
        upper_x = x[fit_mask][-1]
        shown = x <= upper_x
        curve_x = np.logspace(
            np.log10(x[fit_mask][0]), np.log10(upper_x), 200
        )
        curve_y = fit["coefficient"] / curve_x ** fit["exponent"] + fit["floor"]
        axes.loglog(
            curve_x,
            curve_y,
            color=_DARK_COLORS[index],
            zorder=1,
            **_FIT_LINE_STYLE,
        )
        axes.loglog(
            x[shown],
            y[shown],
            linestyle="none",
            marker="o",
            markersize=4.0,
            color=_COLORS[index],
            zorder=2,
            label=rf"{model} (${_legend_exponent(fit)}$)",
        )

    axes.set_xlabel(
        r"Residual aware Training Time $H(\alpha)/\eta(\alpha)^{3/2}$"
    )
    axes.set_ylabel(r"Generalization Error $\epsilon_g$", labelpad=0)
    _add_legend(axes)
    return figure


def _baseline_removed_plot(
    x_values: np.ndarray,
    top1_errors: np.ndarray,
    fits: list[dict[str, float]],
) -> Figure:
    figure = plt.figure(figsize=[3.2, 2.4])
    axes = figure.add_axes([0.2, 0.2, 0.75, 0.75])
    axes.grid(False)
    fit_mask = _window_mask(_NOMINAL_WINDOW)

    for index, ((model, _, _), x, y, fit) in enumerate(
        zip(_MODELS, x_values, top1_errors, fits)
    ):
        residual = y - fit["floor"]
        upper_x = x[fit_mask][-1]
        tolerance = 10000 * np.finfo(float).eps * max(1.0, float(y.max()))
        shown = (x <= upper_x) & (residual > tolerance)
        curve_x = np.logspace(
            np.log10(x[fit_mask][0]), np.log10(upper_x), 200
        )
        curve_y = fit["coefficient"] / curve_x ** fit["exponent"]
        axes.loglog(
            curve_x,
            curve_y,
            color=_DARK_COLORS[index],
            zorder=1,
            **_FIT_LINE_STYLE,
        )
        axes.loglog(
            x[shown],
            residual[shown],
            linestyle="none",
            marker="o",
            markersize=4.0,
            color=_COLORS[index],
            zorder=2,
            label=rf"{model} (${_legend_exponent(fit)}$)",
        )

    axes.set_xlabel(
        r"Residual aware Training Time $H(\alpha)/\eta(\alpha)^{3/2}$"
    )
    axes.set_ylabel(
        r"Generalization Error $\epsilon_g$ $-\,\epsilon_{\rm floor}$",
        labelpad=0,
    )
    _add_legend(axes)
    return figure


def plot_pythia() -> tuple[Figure, Figure]:
    """Create, save, display, and return the two fixed Pythia manuscript plots."""

    top1_errors = _load_top1_errors()
    x_values = _residual_aware_times()
    fits = _fit_models(x_values, top1_errors)
    _configure_plot_style()

    for (model, _, _), fit in zip(_MODELS, fits):
        print(
            f"{model}: exponent={fit['exponent']:.6f}, "
            f"+{fit['exponent_error_plus']:.6f}, "
            f"-{fit['exponent_error_minus']:.6f}, "
            f"floor={fit['floor']:.6f}"
        )

    raw_figure = _raw_plot(x_values, top1_errors, fits)
    baseline_removed_figure = _baseline_removed_plot(x_values, top1_errors, fits)

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _DATA_DIR / "pythia-top1_error-fit.pdf"
    baseline_path = _DATA_DIR / "pythia-top1_error-fit-baseline-removed.pdf"
    raw_figure.savefig(raw_path)
    baseline_removed_figure.savefig(baseline_path)
    if plt.get_backend().lower() != "agg":
        plt.show()
    print(f"Saved {raw_path}")
    print(f"Saved {baseline_path}")
    return raw_figure, baseline_removed_figure


if __name__ == "__main__":
    plot_pythia()
