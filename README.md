# QML DPG Summer School 2026

This repository holds the exercises for the QML Summer School 2026.

For a gentle introduction in this topic, you can find some meaningful information in my [QC Handbook](https://iraitzm.github.io/qc-handbook).

> **New here?** [**Getting started**](GETTING_STARTED.md) walks through cloning the repo,
> setting up the environment with `uv`, running the notebooks with `marimo`, and
> registering in IQM Resonance. Or just click any badge below to run in the browser.
>
> Prefer plain `pip`, e.g. on Windows or a machine without an NVIDIA GPU? Use the
> [`requirements.txt` setup](GETTING_STARTED.md#alternative-pip-and-requirementstxt) instead. But make sure you use Python>=3.12.

## Session 1

All about creating our first implementations for Quantum Computing and understanding key concepts around Quantum Machine Learning.

**Getting started** Learn how quantum computing is developed from scratch and why frameworks like Qiskit can really accelerate the development cycle.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%201/1.1-Getting%20the%20basics.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%201/1.1-Getting%20the%20basics.ipynb)

**Observables and derivatives** One of the main aspects of quantum computing is being able to simulate systems. Evolve states and compute expectation values over observables. Quantum circuits can behave similar to neural networks, defining a parameterized function, computing its derivative and finding the right set of parameters minimizing an energy function. 

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%201/1.2-Going%20variational.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%201/1.2-Going%20variational.ipynb)

**Quantum Machine Learning** A gentle approach to how Quantum Computing evolved towards Quantum Machine Learning, the main categories for kernel based and neural networks inspired models and the challenges one will face defining those models.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%201/1.3-QML.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%201/1.3-QML.ipynb)

## Session 2

Running on hardware using IQM Resonance provided chips Emerald, Garnet and Sirius.

**Getting started** First access and submission of circuits to IQM chips.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%202/2.1-Running%20on%20hardware.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%202/2.1-Running%20on%20hardware.ipynb)

**Handling the noise** By means of error mitigation and cancellation techniques we can improve the results obtained when running on hardware and make it close to what the ideal device would do.

> **Achtung**: Colab notebook may not work due to IPython version showing incompatibilities with IQM's libraries.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%202/2.2-Handling%20noise.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%202/2.2-Handling%20noise.ipynb)

**QML on hardware** We can combine everything on a single notebook that launches QML workloads against the hardware.

> **Achtung**: Watch for credits to be spent, IQM allows for a limited number of credits. Use https://www.iqmacademy.com/qpu/resourceCalculator/

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%202/2.3-QML%20on%20hardware.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%202/2.3-QML%20on%20hardware.ipynb)

## Session 3

How can we use Tensor Networks for better AI and QC. Well, let's explore what's available.

> ITensor CQL 2025 workshop slides and tutorials available as well in https://itensor.org/school/ 

**Getting started** Getting familiar with Tensor Networks in Python. In this example we will get to implement some of the theoretical foundations already seen but also get to know how TNs are used in the domain of ML and QML.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%203/3.1-Tensor%20networks.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%203/3.1-Tensor%20networks.ipynb)

**NNs as TNs** AI and QC have a common ground on matrix operations, they both share the need for efficient matrix multiplication. Given that TNs are meant exactly for this what if we could use this formalism for efficient NN construction.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%203/3.2-Neural%20Networks%20as%20TNs.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%203/3.2-Neural%20Networks%20as%20TNs.ipynb)

**QNNs on TNs** Let's bring all together in this final notebook. We will create a hybrid (classical-quantum) neural network running on top of TN simulators making us of all the frameworks and interoperable pieces we have seen up to now.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Session%203/3.3-QML%20on%20TNs.py)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IraitzM/qml-dpg-2026/blob/main/Session%203/3.3-QML%20on%20TNs.ipynb)

## Extra

As the Summer School progresses we would like to offer some challenges, so under the **Extra** folder, you will find some marimo notebooks for you to exercise the concepts seen during the lectures.

### Boolean penalty Hamiltonian
Build Boolean-logic penalty gadgets (AND, then OR by De Morgan), substitute Pauli operators for Boolean variables to turn a QUBO into an Ising Hamiltonian, tune a penalty weight against a competing objective, and simulate with Qiskit circuits to see which bitstring actually wins.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Extra/boolean_penalty_hamiltonians_exercise.py)

### T-doped circuits
Measure why classical simulation cost tracks a circuit's T-count rather than its qubit or gate count: build T-doped circuits, implement the Bravyi–Gosset stabilizer-rank bound as a formula, and run circuits through Aer's `extended_stabilizer` method to watch the wall-clock cost climb with T-count.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Extra/t_doped_exercise.py)

### Tensor exercises
Contract two tensors by hand and check it against `quimb`, count how many parameters an MPS actually stores as a function of bond dimension $\chi$, decompose a single-qubit unitary into Pauli rotations, and simulate a small circuit as an MPS to read off a measurement probability.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Extra/tensor_networks_exercise.py)

### TN for QML
Implement the single contraction step that *is* an MPS classifier's forward pass, confirm that `default.tensor` is a drop-in replacement for a statevector device on the exact same circuit, and see directly how much entanglement a fixed bond dimension can (and can't) hold, from a GHZ state up to a maximally-entangled one.

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/IraitzM/qml-dpg-2026/blob/main/Extra/tn_for_qml_exercise.py)