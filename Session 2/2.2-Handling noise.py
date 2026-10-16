# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "mitiq==1.0.0",
#     "numpy==2.2.6",
#     "pylatexenc==2.10",
#     "qiskit==2.1.2",
#     "qiskit-aer==0.17.2",
#     "qiskit-ibm-runtime==0.45.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We saw in previous exercise that running against hardware is not perfect, but there are some improvements we can make to compensate such issues.

    As before, set `USE_FAKE = True` to work against a local noisy copy of an IBM device. It is a good idea to go through the whole notebook that way first and only then spend QPU time on the real thing.
    """)
    return


@app.cell
def _():
    # True runs everything on a local noisy copy of an IBM device
    USE_FAKE = False
    return (USE_FAKE,)


@app.cell
def _(USE_FAKE):
    import os
    from getpass import getpass

    if not USE_FAKE and "QISKIT_IBM_TOKEN" not in os.environ:
        os.environ["QISKIT_IBM_TOKEN"] = getpass("Here goes your IBM Quantum API key:")
        os.environ["QISKIT_IBM_INSTANCE"] = getpass("And your instance CRN (leave it empty to use any of yours):")
    return (os,)


@app.cell
def _(USE_FAKE, os):
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.fake_provider import FakeFez

    if USE_FAKE:
        backend = FakeFez()
    else:
        service = QiskitRuntimeService(
            channel="ibm_quantum_platform",
            token=os.environ["QISKIT_IBM_TOKEN"],
            instance=os.environ.get("QISKIT_IBM_INSTANCE") or None,
        )
        # The device with the shortest queue, or pick one by name with service.backend("ibm_fez")
        backend = service.least_busy(operational=True, simulator=False)

    print(f"Connected to : {backend.name}")
    return (backend,)


@app.cell
def _(backend):
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)
    return (sampler,)


@app.cell
def _():
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram

    num_qubits = 7

    large_circ = QuantumCircuit(num_qubits, num_qubits)

    large_circ.h(0)

    for i in range(0, num_qubits-1):
        large_circ.cx(i,i+1)

    large_circ.measure_all(add_bits = False)

    simulator = AerSimulator()

    large_results = simulator.run(large_circ, shots=1000).result()
    local_counts  = large_results.get_counts(large_circ)

    plot_histogram(local_counts)
    return QuantumCircuit, large_circ, local_counts, num_qubits, plot_histogram


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By inspecting the chip we can see that the quality of the qubits is not homogeneous. There might be better results depending which qubit pair you select.
    """)
    return


@app.cell
def _(backend):
    from qiskit.visualization import plot_error_map

    plot_error_map(backend)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus, a qubit selection may be worth.

    ## Qubit selection

    Our GHZ circuit is a chain of CNOTs between neighbours, so it fits perfectly on a line of 7 connected qubits. We can walk the coupling map, list every possible line and score each one by how likely it is that nothing goes wrong on it: all its two-qubit gates and all its measurements succeeding.

    $$
    \text{cost} = 1 - \prod_{q \in \text{line}} (1 - \epsilon_{\text{readout}}(q)) \prod_{(a,b) \in \text{line}} (1 - \epsilon_{\text{CZ}}(a,b))
    $$
    """)
    return


@app.cell
def _(backend, num_qubits):
    import numpy as np

    target = backend.target
    two_qubit_gate = "cz" if "cz" in target.operation_names else "ecr"

    def error(op_name, qargs):
        props = target[op_name].get(qargs)
        if props is None or props.error is None:
            return 1.0
        return props.error

    def edge_error(a, b):
        return min(error(two_qubit_gate, (a, b)), error(two_qubit_gate, (b, a)))

    def get_lines(length):
        graph = backend.coupling_map.graph.to_undirected(multigraph=False)

        def extend(path):
            if len(path) == length:
                yield path
                return
            for neighbour in graph.neighbors(path[-1]):
                if neighbour not in path:
                    yield from extend(path + [neighbour])

        for start in graph.node_indices():
            for path in extend([start]):
                # Each line shows up twice, once from each end
                if path[0] < path[-1]:
                    yield path

    def line_cost(path):
        readout = np.prod([1 - error("measure", (q,)) for q in path])
        gates = np.prod([1 - edge_error(a, b) for a, b in zip(path, path[1:])])
        return 1 - readout * gates

    layouts = sorted(get_lines(num_qubits), key=line_cost)

    for _layout in layouts[:10]:
        print(f"{_layout} -> cost {line_cost(_layout):.4f}")
    return error, layouts, line_cost, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus, when we transpile the circuit we can point out to the layout with minimum cost and compare it with just taking the first qubits of the chip.
    """)
    return


@app.cell
def _(backend, large_circ, layouts, line_cost, num_qubits):
    from qiskit.transpiler import generate_preset_pass_manager

    naive_layout = list(range(num_qubits))
    best_layout = layouts[0]
    print(f"Naive layout cost {line_cost(naive_layout):.4f}, best layout cost {line_cost(best_layout):.4f}")

    def to_isa(circuit, layout):
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            initial_layout=layout,
            optimization_level=3,
            seed_transpiler=42
        )
        return pass_manager.run(circuit)

    # Propose a set of qubits to place the circuit in
    isa_naive = to_isa(large_circ, naive_layout)

    # Use the minimum cost one
    isa_optim = to_isa(large_circ, best_layout)
    return best_layout, isa_naive, isa_optim, to_isa


@app.cell
def _(isa_naive, isa_optim, sampler):
    # Both circuits travel in the same job, less overhead and less waiting
    hw_job = sampler.run([isa_naive, isa_optim], shots=1000)
    print(f"Job ID: {hw_job.job_id()} with status {hw_job.status()}")
    return (hw_job,)


@app.cell
def _(hw_job, plot_histogram):
    hw_results = hw_job.result()
    counts_hw = hw_results[0].data.c.get_counts()
    counts_hw_optim = hw_results[1].data.c.get_counts()

    plot_histogram([counts_hw, counts_hw_optim], legend=["Naive layout", "Min. cost layout"])
    return (counts_hw_optim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we look into the cost we used, readout errors are usually the largest contribution. That is one of the first things we will need to look to compensate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Readout Error Mitigation (REM)

    Redout error refers to the systematic error one can observe when measuring a qubit. By default, each qubit has a percentage or errors and those can be characterized an compensated.

    ![rem](https://mitiq.readthedocs.io/en/latest/_images/rem_workflow.svg)

    [Mitiq](https://mitiq.readthedocs.io/en/stable/#error-mitigation-techniques) is a pretty useful library that will ease the path using these techniques. Depending on where our circuit landed we would need to retrieve its readout error or flipping probabilities.
    """)
    return


@app.cell
def _(best_layout, error):
    readout_errors = {q: error("measure", (q,)) for q in best_layout}
    readout_errors
    return (readout_errors,)


@app.cell
def _(best_layout, np, num_qubits, readout_errors):
    from mitiq.rem import generate_tensored_inverse_confusion_matrix

    def confusion_matrix(p_flip):
        # Same flip probability for 0->1 and 1->0
        return np.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])

    # Qiskit bitstrings read right to left, so the last qubit of the layout goes first
    confusion_matrices = [confusion_matrix(readout_errors[q]) for q in reversed(best_layout)]

    inverse_confusion_matrix = generate_tensored_inverse_confusion_matrix(num_qubits, confusion_matrices)
    inverse_confusion_matrix
    return (inverse_confusion_matrix,)


@app.cell
def _(counts_hw_optim, inverse_confusion_matrix, local_counts, plot_histogram):
    from mitiq import MeasurementResult
    from mitiq.rem.inverse_confusion_matrix import mitigate_measurements

    hw_optim_results = MeasurementResult.from_counts(counts_hw_optim)

    mitigated_result = mitigate_measurements(hw_optim_results, inverse_confusion_matrix)
    counts_hw_optim_rem = mitigated_result.get_counts()

    plot_histogram([local_counts, counts_hw_optim_rem], legend=["Ideal", "Min. cost layout + REM"])
    return MeasurementResult, mitigate_measurements


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Depth optimization

    Sometimes it is as simple as rearranging some of the gates so that the overall depth gets diminished.
    """)
    return


@app.cell
def _(isa_optim):
    isa_optim.draw('mpl', fold=-1, idle_wires=False)
    return


@app.cell
def _(QuantumCircuit, best_layout, num_qubits, to_isa):
    compact = QuantumCircuit(num_qubits, num_qubits)

    # Start from the middle and spread the entanglement both ways
    compact.h(4)

    for _i in range(4, 0, -1):
        compact.cx(_i, _i-1)

    for _i in range(4, num_qubits-1):
        compact.cx(_i, _i+1)

    compact.measure_all(add_bits=False)

    isa_compact = to_isa(compact, best_layout)

    isa_compact.draw('mpl', fold=-1, idle_wires=False)
    return (isa_compact,)


@app.cell
def _(isa_compact, isa_optim):
    print(f"Naive version has {isa_optim.depth()} depth")
    print(f"Compact version has {isa_compact.depth()} depth")
    return


@app.cell
def _(isa_compact, sampler):
    hw_compact_job = sampler.run([isa_compact], shots=1000)
    print(f"Job ID: {hw_compact_job.job_id()} with status {hw_compact_job.status()}")
    return (hw_compact_job,)


@app.cell
def _(
    MeasurementResult,
    hw_compact_job,
    inverse_confusion_matrix,
    local_counts,
    mitigate_measurements,
    plot_histogram,
):
    counts_hw_compact = hw_compact_job.result()[0].data.c.get_counts()

    hw_compact_results = MeasurementResult.from_counts(counts_hw_compact)

    mitigated_compact_result = mitigate_measurements(hw_compact_results, inverse_confusion_matrix)
    counts_compact_rem = mitigated_compact_result.get_counts()

    plot_histogram([local_counts, counts_compact_rem], legend=["Ideal", "Compact + REM"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dynamical Decoupling (DD)

    When qubits are idling they do accumulate error, so a known technique to mitigate this is to apply operations that do contradict one another. Even though the effect of applying to consecutive XX gates is none, the fact that operations are happening in those idling qubits while waiting reduces the amount or error being accumulated.

    One thing we did not mention before is that gates also do not exist. These are simply discretized analog operations (microwave pulses) over the hardware, and each of them takes some time. IBM no longer exposes those pulses to users, but every native gate has a known duration in the `target`, so the transpiler can **schedule** the circuit: decide when each operation starts and fill the gaps with `delay` instructions.
    """)
    return


@app.cell
def _(backend):
    for _gate in ["sx", "x", "cz", "measure"]:
        if _gate in backend.target.operation_names:
            _duration = next(p.duration for p in backend.target[_gate].values() if p is not None and p.duration)
            print(f"{_gate:>8}: {_duration * 1e9:.0f} ns")
    return


@app.cell
def _(backend, isa_compact):
    from qiskit.circuit.library import XGate
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import (
        ALAPScheduleAnalysis,
        PadDelay,
        PadDynamicalDecoupling,
    )

    def drop_idle_delays(circuit):
        # The scheduler also pads the ~150 qubits we do not use, keep only our wires
        active = {
            q
            for instruction in circuit.data
            if instruction.operation.name not in ("delay", "barrier")
            for q in instruction.qubits
        }
        clean = circuit.copy_empty_like()
        for instruction in circuit.data:
            if instruction.operation.name == "delay" and not set(instruction.qubits) & active:
                continue
            clean.append(instruction)
        return clean

    schedule = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDelay(target=backend.target),
    ])
    scheduled = drop_idle_delays(schedule.run(isa_compact))

    scheduled.draw('mpl', fold=-1, idle_wires=False)
    return (
        ALAPScheduleAnalysis,
        PadDynamicalDecoupling,
        PassManager,
        XGate,
        drop_idle_delays,
        scheduled,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Those wait times add noise to the whole execution. Let's fill them with pairs of X gates, which together do nothing but keep the qubits busy.
    """)
    return


@app.cell
def _(
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    PassManager,
    XGate,
    backend,
    drop_idle_delays,
    isa_compact,
):
    dd_schedule = PassManager([
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(target=backend.target, dd_sequence=[XGate(), XGate()]),
    ])
    scheduled_dd = drop_idle_delays(dd_schedule.run(isa_compact))

    scheduled_dd.draw('mpl', fold=-1, idle_wires=False)
    return (scheduled_dd,)


@app.cell
def _(sampler, scheduled, scheduled_dd):
    dd_job = sampler.run([scheduled, scheduled_dd], shots=1000)
    print(f"Job ID: {dd_job.job_id()} with status {dd_job.status()}")
    return (dd_job,)


@app.cell
def _(dd_job, plot_histogram):
    dd_results = dd_job.result()
    counts_scheduled = dd_results[0].data.c.get_counts()
    counts_dd = dd_results[1].data.c.get_counts()

    plot_histogram([counts_scheduled, counts_dd], legend=["Scheduled", "Scheduled + DD"])
    return (counts_dd,)


@app.cell
def _(
    MeasurementResult,
    counts_dd,
    inverse_confusion_matrix,
    local_counts,
    mitigate_measurements,
    plot_histogram,
):
    dd_measurements = MeasurementResult.from_counts(counts_dd)

    mitigated_dd_results = mitigate_measurements(dd_measurements, inverse_confusion_matrix)
    counts_dd_rem = mitigated_dd_results.get_counts()

    plot_histogram([local_counts, counts_dd_rem], legend=["Ideal", "Compact + DD + REM"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > With `USE_FAKE = True` do not expect much of a difference here. The local noise model has no crosstalk or coherent errors, which is precisely what DD is best at suppressing.

    Doing it by hand is good to understand what happens, but IBM's primitives can also do it for us at execution time:

    ```python
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"  # or "XX", "XpXm"
    ```

    ## Zero Noise Extrapolation (ZNE)

    So far we looked at bitstrings. Many algorithms (and QML models) only need expectation values, and for those IBM's **Estimator** primitive comes with error mitigation built in. ZNE amplifies the noise on purpose, running the circuit at noise levels $\lambda = 1, 3, 5$ (by folding gates as $G G^\dagger G$), and extrapolates the trend back to $\lambda = 0$.

    For a GHZ state $\langle Z_0 Z_i \rangle = 1$ for every qubit $i$, so we know exactly what the ideal answer is.
    """)
    return


@app.cell
def _(QuantumCircuit, best_layout, num_qubits, to_isa):
    from qiskit.quantum_info import SparsePauliOp

    # Estimator circuits do not have measurements, those come from the observables
    ghz = QuantumCircuit(num_qubits)
    ghz.h(0)
    for _i in range(num_qubits - 1):
        ghz.cx(_i, _i + 1)

    isa_ghz = to_isa(ghz, best_layout)

    # Z on qubit 0 and qubit i (Pauli labels also read right to left)
    observables = [
        SparsePauliOp("I" * (num_qubits - 1 - _i) + "Z" + "I" * (_i - 1) + "Z").apply_layout(isa_ghz.layout)
        for _i in range(1, num_qubits)
    ]
    return isa_ghz, observables


@app.cell
def _(USE_FAKE, backend, isa_ghz, observables):
    from qiskit_ibm_runtime import EstimatorV2 as Estimator

    # No mitigation at all
    estimator_raw = Estimator(mode=backend)
    estimator_raw.options.resilience_level = 0
    estimator_raw.options.default_shots = 1000

    # Same, with ZNE on top
    estimator_zne = Estimator(mode=backend)
    estimator_zne.options.resilience_level = 0
    estimator_zne.options.default_shots = 1000
    estimator_zne.options.resilience.zne_mitigation = True
    estimator_zne.options.resilience.zne.noise_factors = (1, 3, 5)
    estimator_zne.options.resilience.zne.extrapolator = ("exponential", "linear")

    if USE_FAKE:
        print("Local fake backends ignore mitigation options, both runs will look the same")

    raw_job = estimator_raw.run([(isa_ghz, observables)])
    zne_job = estimator_zne.run([(isa_ghz, observables)])
    print(f"Job IDs: {raw_job.job_id()} and {zne_job.job_id()}")
    return raw_job, zne_job


@app.cell
def _(np, num_qubits, raw_job, zne_job):
    import matplotlib.pyplot as plt

    raw_evs = raw_job.result()[0].data
    zne_evs = zne_job.result()[0].data

    _x = np.arange(1, num_qubits)
    plt.figure(figsize=(10, 5))
    plt.axhline(1.0, color="black", linestyle="--", label="Ideal")
    plt.errorbar(_x - 0.1, raw_evs.evs, yerr=raw_evs.stds, fmt="o", capsize=4, label="Raw")
    plt.errorbar(_x + 0.1, zne_evs.evs, yerr=zne_evs.stds, fmt="o", capsize=4, label="ZNE")
    plt.xticks(_x, [f"$Z_0 Z_{{{i}}}$" for i in _x])
    plt.ylabel("Expectation value")
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The further the qubit is from qubit 0 the more gates sit in between, and the lower the raw value gets. ZNE pulls them back towards 1, at the price of larger error bars and three times the circuits.

    There are additional techniques worth covering:

    * **Twirled Readout Error eXtinction (TREX)**, IBM's own readout mitigation, enabled with `resilience_level = 1` (the Estimator default).
    * **Probabilistic error cancellation (PEC)** that uses a set of noise gates as linear combinations to construct ideal noiseless gates. Also available in the Estimator (`resilience.pec_mitigation = True`), but it is expensive in QPU time, so better keep it away from the Open Plan.

    ![pec](https://mitiq.readthedocs.io/en/latest/_images/pec_workflow2_steps.png)
    """)
    return


if __name__ == "__main__":
    app.run()
