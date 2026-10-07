"""Minimal reproduction of the MNIST experiment used in the manuscript."""

from __future__ import annotations

import math
import random
from pathlib import Path
from zipfile import BadZipFile

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / "data"
_CACHE_PATH = _DATA_DIR / "mnist_histories.npz"
_FIGURE_DIR = _HERE / "figures"
_STYLE_PATH = _HERE / "plt_style.pstyle"

_SEEDS = (0, 1, 2, 3, 4, 5)
_TRAIN_SIZE = 60_000
_TEST_SIZE = 10_000
_EPOCHS = 4
_TOTAL_STEPS = _TRAIN_SIZE * _EPOCHS
_EVAL_POINTS = 32
_ALPHA_NORMALIZER = 784.0
_LEARNING_RATE = 0.01
_FIT_STEP_MIN = 6_580
_FIT_STEP_MAX = 240_000
_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081


def _evaluation_steps(total_steps: int = _TOTAL_STEPS) -> np.ndarray:
    log_steps = np.unique(
        np.rint(np.geomspace(1, total_steps, num=_EVAL_POINTS)).astype(np.int64)
    )
    epoch_ends = range(_TRAIN_SIZE, total_steps + 1, _TRAIN_SIZE)
    return np.asarray(sorted({0, *log_steps.tolist(), *epoch_ends}), dtype=np.int64)


_STEPS = _evaluation_steps()


class _EpochMNIST(Dataset):
    """MNIST tensors with deterministic augmentation from epoch two onward."""

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        *,
        seed: int,
        epoch: int,
    ) -> None:
        self.images = images
        self.labels = labels
        self.seed = seed
        self.epoch = epoch

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index].to(torch.float32).unsqueeze(0).div_(255.0)

        if self.epoch >= 2:
            generator = torch.Generator(device="cpu")
            augmentation_seed = (
                self.seed * 1_000_003
                + self.epoch * 97_409
                + int(index) * 65_537
                + 17
            ) % (2**63 - 1)
            generator.manual_seed(augmentation_seed)
            angle = float((2.0 * torch.rand((), generator=generator).item() - 1.0) * 10.0)
            shift_x = int(torch.randint(-1, 2, (1,), generator=generator).item())
            shift_y = int(torch.randint(-1, 2, (1,), generator=generator).item())
            image = TVF.affine(
                image,
                angle=angle,
                translate=[shift_x, shift_y],
                scale=1.0,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

        image.sub_(_MNIST_MEAN).div_(_MNIST_STD)
        return image.contiguous(), self.labels[index]


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model(seed: int, device: torch.device) -> nn.Module:
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    return model.to(device)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_examples += int(labels.numel())
    model.train()
    return 1.0 - total_correct / total_examples, total_loss / total_examples


def _train_seed(
    seed: int,
    device: torch.device,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
    steps: np.ndarray = _STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Train one seed; ``steps`` is injectable for a short smoke test."""

    _set_seed(seed)
    pin_memory = device.type == "cuda"
    test_loader = DataLoader(
        _EpochMNIST(test_images, test_labels, seed=seed, epoch=1),
        batch_size=1024,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    model = _make_model(seed, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE)

    checkpoint_indices = {int(step): index for index, step in enumerate(steps)}
    test_error = np.full(len(steps), np.nan, dtype=np.float64)
    test_loss = np.full(len(steps), np.nan, dtype=np.float64)

    test_error[checkpoint_indices[0]], test_loss[checkpoint_indices[0]] = _evaluate(
        model, test_loader, device
    )
    max_step = int(steps[-1])
    update = 0

    for epoch in range(1, _EPOCHS + 1):
        if update >= max_step:
            break
        shuffle_generator = torch.Generator(device="cpu")
        shuffle_generator.manual_seed(
            (seed + (epoch - 1) * 1_000_003) % (2**63 - 1)
        )
        train_loader = DataLoader(
            _EpochMNIST(train_images, train_labels, seed=seed, epoch=epoch),
            batch_size=1,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=0,
            pin_memory=pin_memory,
        )

        for images, labels in train_loader:
            if update >= max_step:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at update {update + 1}")
            loss.backward()
            optimizer.step()
            update += 1

            checkpoint_index = checkpoint_indices.get(update)
            if checkpoint_index is not None:
                test_error[checkpoint_index], test_loss[checkpoint_index] = _evaluate(
                    model, test_loader, device
                )

    if update != max_step or not np.all(np.isfinite(test_error + test_loss)):
        raise RuntimeError("MNIST training did not reach every requested checkpoint")
    return test_error, test_loss


def _empty_results() -> dict[str, np.ndarray]:
    return {
        "seeds": np.empty(0, dtype=np.int64),
        "steps": _STEPS.copy(),
        "test_error": np.empty((0, len(_STEPS)), dtype=np.float64),
        "test_loss": np.empty((0, len(_STEPS)), dtype=np.float64),
    }


def _validate_results(results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    raw_seeds = np.asarray(results["seeds"])
    raw_steps = np.asarray(results["steps"])
    if raw_seeds.dtype.kind not in "iu" or raw_steps.dtype.kind not in "iu":
        raise ValueError("cached seeds and steps must be integer arrays")
    seeds = raw_seeds.astype(np.int64, copy=False)
    steps = raw_steps.astype(np.int64, copy=False)
    test_error = np.asarray(results["test_error"], dtype=np.float64)
    test_loss = np.asarray(results["test_loss"], dtype=np.float64)

    if seeds.ndim != 1 or not np.array_equal(seeds, np.unique(seeds)):
        raise ValueError("cached seeds must be sorted and unique")
    if not set(seeds.tolist()).issubset(_SEEDS):
        raise ValueError("cache contains unexpected seeds")
    if not np.array_equal(steps, _STEPS):
        raise ValueError("cache contains an unexpected evaluation-step grid")
    expected_shape = (len(seeds), len(_STEPS))
    if test_error.shape != expected_shape or test_loss.shape != expected_shape:
        raise ValueError("cached metric arrays have unexpected shapes")
    if len(seeds) and (
        not np.all(np.isfinite(test_error))
        or not np.all(np.isfinite(test_loss))
        or np.any(test_error <= 0)
        or np.any(test_error > 1)
        or np.any(test_loss <= 0)
    ):
        raise ValueError("cached metrics are outside their valid ranges")
    return {
        "seeds": seeds,
        "steps": steps,
        "test_error": test_error,
        "test_loss": test_loss,
    }


def _load_cache() -> dict[str, np.ndarray]:
    if not _CACHE_PATH.is_file():
        return _empty_results()
    try:
        with np.load(_CACHE_PATH, allow_pickle=False) as cache:
            if set(cache.files) != {"seeds", "steps", "test_error", "test_loss"}:
                raise ValueError("cache contains unexpected fields")
            results = {name: cache[name].copy() for name in cache.files}
        return _validate_results(results)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
        raise ValueError(
            f"Invalid MNIST cache at {_CACHE_PATH}; rerun with force=True"
        ) from exc


def _save_cache(results: dict[str, np.ndarray]) -> None:
    results = _validate_results(results)
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _CACHE_PATH.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_path, **results)
    temporary_path.replace(_CACHE_PATH)


def _pack_results(
    completed: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    seeds = np.asarray(sorted(completed), dtype=np.int64)
    return {
        "seeds": seeds,
        "steps": _STEPS.copy(),
        "test_error": np.stack([completed[int(seed)][0] for seed in seeds]),
        "test_loss": np.stack([completed[int(seed)][1] for seed in seeds]),
    }


def _load_or_train(*, force: bool, device: str) -> dict[str, np.ndarray]:
    cached = _empty_results() if force else _load_cache()
    completed = {
        int(seed): (cached["test_error"][index], cached["test_loss"][index])
        for index, seed in enumerate(cached["seeds"])
    }
    missing_seeds = [seed for seed in _SEEDS if seed not in completed]
    if not missing_seeds:
        print(f"Loaded all MNIST histories from {_CACHE_PATH}")
        return cached

    train_data = MNIST(root=str(_DATA_DIR), train=True, download=True)
    test_data = MNIST(root=str(_DATA_DIR), train=False, download=True)
    if len(train_data) != _TRAIN_SIZE or len(test_data) != _TEST_SIZE:
        raise RuntimeError("Unexpected MNIST train/test split sizes")
    training_device = _resolve_device(device)

    for seed in missing_seeds:
        print(f"Training MNIST seed {seed} on {training_device} ...")
        completed[seed] = _train_seed(
            seed,
            training_device,
            train_data.data,
            train_data.targets,
            test_data.data,
            test_data.targets,
        )
        cached = _pack_results(completed)
        _save_cache(cached)
        print(f"Saved {len(completed)}/{len(_SEEDS)} seeds to {_CACHE_PATH}")
    return cached


def _fit_power_law(
    steps: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    fit_mask = (steps >= _FIT_STEP_MIN) & (steps <= _FIT_STEP_MAX)
    fit_alpha = steps[fit_mask] / _ALPHA_NORMALIZER
    fit_mean = values[:, fit_mask].mean(axis=0)
    if len(fit_alpha) < 3 or np.any(fit_mean <= 0):
        raise ValueError("Insufficient positive data in the MNIST fit window")

    coefficients, covariance = np.polyfit(
        np.log(fit_alpha), np.log(fit_mean), deg=1, cov=True
    )
    exponent = float(-coefficients[0])
    exponent_standard_error = float(math.sqrt(covariance[0, 0]))
    line_alpha = np.geomspace(fit_alpha[0], fit_alpha[-1], 150)
    line_values = np.exp(np.polyval(coefficients, np.log(line_alpha)))
    return line_alpha, line_values, exponent, exponent_standard_error


def _configure_plot_style() -> None:
    if not _STYLE_PATH.is_file():
        raise FileNotFoundError(f"Could not find Matplotlib style: {_STYLE_PATH}")
    plt.style.use(str(_STYLE_PATH))
    plt.rcParams.update(
        {
            "font.size": 7.8,
            "figure.titlesize": 9,
            "legend.fontsize": 7,
            "axes.titlesize": 9,
            "axes.labelsize": 8.8,
            "figure.figsize": [2.55, 1.7],
            "mathtext.fontset": "cm",
        }
    )


def _plot_metric(
    results: dict[str, np.ndarray],
    metric: str,
    ylabel: str,
    filename: str,
) -> Figure:
    steps = results["steps"]
    values = results[metric]
    positive = steps > 0
    alpha = steps[positive] / _ALPHA_NORMALIZER
    alpha_grid = np.geomspace(alpha[0], alpha[-1], 150)
    interpolated = np.vstack(
        [np.interp(alpha_grid, alpha, seed_values[positive]) for seed_values in values]
    )

    figure = plt.figure(figsize=[2.55, 1.7])
    axes = figure.add_axes([0.17, 0.2, 0.80, 0.78])
    axes.grid(False)
    color = "#0C5DA5"
    axes.loglog(alpha_grid, interpolated.mean(axis=0), color=color)
    axes.fill_between(
        alpha_grid,
        interpolated.min(axis=0),
        interpolated.max(axis=0),
        color=color,
        alpha=0.5,
        edgecolor="none",
    )

    fit_alpha, fit_values, exponent, standard_error = _fit_power_law(steps, values)
    (fit_line,) = axes.loglog(
        fit_alpha,
        fit_values,
        ls="--",
        dashes=[2, 1.8],
        linewidth=2.0,
        color="#003A6D",
        label=rf"$\alpha^{{-({exponent:.3f}\pm{2.0 * standard_error:.3f})}}$",
    )
    axes.set_xlabel(r"Training Time $\alpha$")
    axes.set_ylabel(ylabel, labelpad=0)
    axes.legend(
        handles=[fit_line],
        loc="best",
        handlelength=1.0,
        labelspacing=0.1,
        frameon=False,
        borderpad=0.1,
        borderaxespad=0.7,
        handletextpad=0.5,
        fontsize=8,
    )

    _FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    destination = _FIGURE_DIR / f"{filename}.pdf"
    figure.savefig(destination)
    figure.savefig(destination.with_suffix(".png"), dpi=450)
    return figure


def run_mnist(*, force: bool = False, device: str = "auto") -> tuple[Figure, Figure]:
    """Load or run the fixed MNIST experiment and save its two manuscript plots.

    The first call downloads MNIST, trains any missing seeds, and stores one
    compact cache in ``data/mnist_histories.npz``. Later calls reuse that cache.
    Set ``force=True`` to retrain all six seeds.
    """

    results = _load_or_train(force=force, device=device)
    if not np.array_equal(results["seeds"], np.asarray(_SEEDS)):
        raise RuntimeError("The MNIST result cache is incomplete")
    _configure_plot_style()
    error_figure = _plot_metric(
        results,
        "test_error",
        r"Generalization Error $\epsilon_g$",
        "mnist_test_error_fit_2se",
    )
    loss_figure = _plot_metric(
        results,
        "test_loss",
        r"Test Loss $\mathcal{L}_g$",
        "mnist_test_loss_fit_2se",
    )
    return error_figure, loss_figure


__all__ = ["run_mnist"]
