[![Paper](https://img.shields.io/badge/paper-arXiv%3A2610._-B31B1B.svg)](https://arxiv.org/abs/2610._)

# A Boundary-Layer Mechanism for One-Third Scaling in Online Softmax Classification

Marcel Kühn, Yoon Thelge, Bernd Rosenow

[arXiv:2610._](https://arxiv.org/abs/2610._)

### Abstract
Hard-label classification is usually trained with smooth surrogate losses such as softmax cross-entropy. We isolate a mechanism by which this mismatch between smooth surrogate and discrete labels produces power-law learning curves for a one-layer softmax student trained by online SGD on Gaussian inputs with argmax teacher labels. In the thermodynamic limit, the dynamics reduce to a growing student-teacher alignment and a residual student variance sustained by SGD noise. At late times, only shrinking layers of examples around teacher decision boundaries remain active. This yields an $\alpha^{-1/3}$ law in training time $\alpha$ not only for the test loss but also for the generalization error, one minus test accuracy. The error law is an online-noise effect. The loss law persists without gradient noise. Learning-rate annealing separates the two exponents, improving the error exponent from 1/3 toward 1/2 while slowing the loss. Simulations support the predictions. Correlated Gaussian inputs and whitened pretrained features change transients, not the late-time decay. Experiments with deep neural networks can show compatible error scaling. The mechanism is asymptotic and complementary to spectral explanations and data-statistics explanations of neural scaling laws.


### Quick Start

Run Jupyter from this directory and open **`example_use.ipynb`**. The notebook consolidates the i.i.d. and correlated Gaussian experiments, minibatch-size sweep, direct mechanism diagnostics, MNIST experiment, CIFAR-5M/ViT experiment, Pythia checkpoint analysis. Some Gaussian settings are reduced for illustration.

### Gaussian Experiments

* **`util_iid_gaussian.py`** implements the i.i.d. Gaussian simulations, parameter sweeps, theory curves, and plots.
* **`util_correlated_gaussian.py`** implements experiments with power-law-correlated Gaussian inputs.
* **`util_batch_size.py`** compares the dynamics at several minibatch sizes.
* **`util_mechanism.py`** produces the conditional-error and alignment-gradient diagnostics of the boundary-layer mechanism.

These experiments write reusable NumPy caches and figures under **`data/`**, **`data_K_class/`**, **`figures/`**, and **`figure_K_class/`**.

### Image Experiments

* **`util_mnist.py`** downloads MNIST on first use, runs the fixed six-seed experiment, caches it as **`data/mnist_histories.npz`**, and saves the two manuscript plots under **`figures/`**.
* **`cifar5m_vit_features.py`** extracts ViT-B/16 features from CIFAR-5M. Place the CIFAR-5M `.npz` shards in **`cifar5m_data/`** before running it.
* **`train_cifar5m.py`** trains the teacher and student classifiers on those features.
* **`plot_cifar5m.py`** generates the CIFAR-5M generalization-error figures.

The complete CIFAR-5M command-line workflow used by the notebook is:

```bash
python cifar5m_vit_features.py
python train_cifar5m.py --data_file cifar5m_vit_features.pt --lr 0.0003 --seeds 0 1 2 3 --num_examples 1800000 --test_split 0.01 --use_whitening --bs 1
python plot_cifar5m.py
```


### Pythia Experiment

* **`evaluate_pythia.py`** evaluates one Pythia model at the 13 checkpoint steps on a pinned FineWeb stream and writes a compact result to **`pythia/`**.
* **`plot_pythia.py`** fits the top-1 error curves and saves the raw and baseline-removed manuscript figures.

Generate all three model results before plotting:

```bash
python evaluate_pythia.py 410m
python evaluate_pythia.py 2.8b
python evaluate_pythia.py 12b
python plot_pythia.py
```
