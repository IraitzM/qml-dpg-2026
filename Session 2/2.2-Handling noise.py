# /// script
# dependencies = [
#     "ipython==9.14.1",
#     "iqm-client[qiskit]==34.0.3",
#     "iqm-qubit-selector==1.0.3",
#     "marimo",
#     "matplotlib==3.11.0",
#     "mitiq==1.0.0",
#     "numpy==2.2.6",
#     "qiskit==2.1.2",
#     "qiskit-aer==0.17.2",
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
    """)
    return


@app.cell
def _():
    import os
    from getpass import getpass

    os.environ["IQM_TOKEN"] = getpass("Here goes your IQM API token:")
    return


@app.cell
def _():
    from iqm.qiskit_iqm import IQMProvider

    IQM_URL = "https://resonance.meetiqm.com/"
    QUANTUM_COMPUTER = "garnet"

    provider = IQMProvider(IQM_URL, quantum_computer=QUANTUM_COMPUTER)
    backend = provider.get_backend()
    print(f"Connected to : {backend.name}")
    return IQM_URL, QUANTUM_COMPUTER, backend


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

    ![garnet](public/image_1.png)

    Thus, a qubit selection may be worth.

    ## Qubit selection
    """)
    return


@app.cell
def _(backend, large_circ):
    from iqm.qubit_selector.qubit_selector import CostEvaluator, ReadoutMode

    layouts_with_readout, cost_with_readout = CostEvaluator(backend, large_circ, readoutmode=ReadoutMode.FIDELITY).get_top_layouts(num_layouts=10)

    cost_with_readout
    return (layouts_with_readout,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus, when we transpile the circuit we can point out to the layout with minimum cost.
    """)
    return


@app.cell
def _(backend, large_circ, layouts_with_readout, num_qubits):
    from qiskit import transpile

    # Propose a set of qubits to place the circuit in
    circ_iqm = transpile(
        large_circ,
        backend=backend,
        initial_layout=list(range(num_qubits)),
        optimization_level=3,
        seed_transpiler=42
    )

    # Use the minimum cost one
    circ_iqm_optim = transpile(
        large_circ,
        backend=backend,
        initial_layout=layouts_with_readout[0],
        optimization_level=3,
        seed_transpiler=42
    )
    return circ_iqm, circ_iqm_optim, transpile


@app.cell
def _(backend, circ_iqm, circ_iqm_optim):
    hw_job = backend.run(circ_iqm, shots=1000)
    hw_optim_job = backend.run(circ_iqm_optim, shots=1000)
    return hw_job, hw_optim_job


@app.cell
def _(hw_job, hw_optim_job, plot_histogram):
    counts_hw = hw_job.result().get_counts()
    counts_hw_optim = hw_optim_job.result().get_counts()

    plot_histogram([counts_hw, counts_hw_optim], legend=["Random layout", "Min. cost layout"])

    return (counts_hw_optim,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we look into the optimization process, we can see that it mainly focuses on the qubits with minimum redout error. That is one of the first things we will need to look to compensate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Readout Error Mitigation (REM)

    Redout error refers to the systematic error one can observe when measuring a qubit. By default, each qubit has a percentage or errors and those can be caracteried an compensated.

    ![rem](https://mitiq.readthedocs.io/en/latest/_images/rem_workflow.svg)

    [Mitiq](https://mitiq.readthedocs.io/en/stable/#error-mitigation-techniques) is a pretty useful library that will ease the path using these techniques. Depending on where our cirquit landed we would need to retrieve its readout error or flipping probabilities.
    """)
    return


@app.cell
def _(backend):
    from iqm.qubit_selector.qubit_selector import CalibrationDataManager

    calibration_data = CalibrationDataManager().get_calibration_fidelities(backend)
    calibration_data['readout']
    return (calibration_data,)


@app.cell
def _(calibration_data, num_qubits):
    import numpy as np
    from mitiq.rem import generate_inverse_confusion_matrix

    p_flip = 1 - np.mean(list(calibration_data['readout'].values()))

    inverse_confusion_matrix = generate_inverse_confusion_matrix(num_qubits, p_flip, p_flip)
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
def _(circ_iqm):
    circ_iqm.draw('mpl', fold=-1)
    return


@app.cell
def _(QuantumCircuit, backend, layouts_with_readout, num_qubits, transpile):
    compact = QuantumCircuit(num_qubits, num_qubits)

    for _i in range(num_qubits):
        compact.h(4)

    for _i in range(4, 0, -1):
        compact.cx(_i, _i-1)

    for _i in range(4, num_qubits-1):
        compact.cx(_i, _i+1)

    compact.measure_all(add_bits=False)

    compact_iqm = transpile(
        compact,
        backend=backend,
        initial_layout=layouts_with_readout[0],
        optimization_level=3,
        seed_transpiler=42
    )

    compact_iqm.draw('mpl', fold=-1)
    return (compact_iqm,)


@app.cell
def _(circ_iqm, compact_iqm):
    print(f"Naive version has {circ_iqm.depth()} depth")
    print(f"Compact version has {compact_iqm.depth()} depth")
    return


@app.cell
def _(backend, compact_iqm):
    hw_compact_job = backend.run(compact_iqm, shots=1000)
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
    counts_hw_compact = hw_compact_job.result().get_counts()

    hw_compact_results = MeasurementResult.from_counts(counts_hw_compact)

    mitigated_compact_result = mitigate_measurements(hw_compact_results, inverse_confusion_matrix)
    counts_compact_rem = mitigated_compact_result.get_counts()

    plot_histogram([local_counts, counts_compact_rem], legend=["Ideal", "Compact + REM"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dynamical Decoupling (DD)

    When qubits are ideling they do accumulate error, so a known technique to mitigate this is to apply operations that do contradict one another. Even though the effect of applying to consecutive XX gates is none, the fact that operations are happening in those idling qubits while waiting reduces the amount or error being accumulated.

    IQM Pulla is IQM's pulse level compiler. One thing we did not mention before is that gates also do not exists. These are simply discretized analog operations over the hardware. Thus, one can access the actual schedule a circuit encodes at such low-level regime.
    """)
    return


@app.cell
def _(IQM_URL, QUANTUM_COMPUTER, backend, compact_iqm):
    from iqm.pulla.pulla import Pulla
    from iqm.pulla.utils_qiskit import qiskit_to_pulla
    from iqm.pulse.playlist.visualisation.base import inspect_playlist
    from IPython.core.display import HTML

    pulla = Pulla(IQM_URL, quantum_computer=QUANTUM_COMPUTER)

    p_circs, compiler = qiskit_to_pulla(pulla, backend, compact_iqm)
    # Manual pulse compilation flow
    job_definition, context = compiler.compile(p_circs)
    playlist = job_definition.sweep_definition.playlist

    HTML(inspect_playlist(playlist, [0]))
    return (
        HTML,
        compiler,
        context,
        inspect_playlist,
        job_definition,
        p_circs,
        pulla,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Those wait times add noise to the whole execution. Let's try adding some arbitrary operations here and there.
    """)
    return


@app.cell
def _(HTML, compiler, inspect_playlist, p_circs):
    from iqm.station_control.interface.models import DDStrategy

    dd_settings = compiler.get_settings(circuits=p_circs)

    dd_strategy = DDStrategy( 
        gate_sequences=[(9, "XYXYYXYX", "asap"), (5, "YXYX", "asap"), (2, "XX", "center")],
        target_qubits=["QB14","QB9","QB15","QB20"],                                  
        )
    dd_settings.stages.dynamical_decoupling.apply_dd_strategy.dd_is_disabled = False

    job_definition_dd, context_dd = compiler.compile(circuits=p_circs, settings=dd_settings, context={"dd_strategy": dd_strategy})
    playlist_dd = job_definition_dd.sweep_definition.playlist

    HTML(inspect_playlist(playlist_dd, [0]))
    return context_dd, job_definition_dd


@app.cell
def _(context, context_dd, job_definition, job_definition_dd, pulla):
    pulla_job = pulla.submit_playlist(job_definition, context=context)
    pulla_dd_job = pulla.submit_playlist(job_definition_dd, context=context_dd)
    return pulla_dd_job, pulla_job


@app.cell
def _(plot_histogram, pulla_dd_job, pulla_job):
    from iqm.pulla.utils_qiskit import sweep_job_to_qiskit

    qiskit_result = sweep_job_to_qiskit(
        pulla_job, shots=1000
    )
    qiskit_result_dd = sweep_job_to_qiskit(
        pulla_dd_job, shots=1000
    )
    counts_hw_pulla = qiskit_result.get_counts()
    counts_dd_pulla = qiskit_result_dd.get_counts()

    plot_histogram([counts_hw_pulla, counts_dd_pulla], legend=["Pulla", "Pulla + DD"])
    return (counts_dd_pulla,)


@app.cell
def _(
    MeasurementResult,
    counts_dd_pulla,
    inverse_confusion_matrix,
    local_counts,
    mitigate_measurements,
    plot_histogram,
):
    pulla_dd_results = MeasurementResult.from_counts(counts_dd_pulla)

    mitigated_pull_dd_results = mitigate_measurements(pulla_dd_results, inverse_confusion_matrix)
    counts_pulla_dd_rem = mitigated_pull_dd_results.get_counts()

    plot_histogram([local_counts, counts_pulla_dd_rem], legend=["Ideal", "Pulla + DD + REM"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are additional techniques worth coverting:

    * **Zero Noise Extrapolation (ZNE)** that amplifies noise in order to understand the trend and thus extrapolate to zero noise expectation values.
    * **Probabilistic error cancellation (PEC)** that uses a set of noise gates as linear combinations to construct ideal noiseless gates.

    ![pec](https://mitiq.readthedocs.io/en/latest/_images/pec_workflow2_steps.png)
    """)
    return


if __name__ == "__main__":
    app.run()
