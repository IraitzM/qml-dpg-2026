# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "pylatexenc==2.10",
#     "qiskit==2.1.2",
#     "qiskit-aer==0.17.2",
#     "iqm-client[qiskit]==34.0.3",
#     "iqm-qubit-selector==1.0.3"
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
    Running circuits locally emulated works for a while until the circuit size grows and the local resources cannot handle those any more. Moreover, to explore what hardware can do and have a real understanding on what those devices can do for us, you better start exploring the usage of real quantum hardware trough some of the many providers out there.

    # IQM Resonance

    [IQM Resonance](https://iqm.tech/products/iqm-resonance/) is the cloud based access to IQM's devices (Emerald, Garnet and Sirius)

    ![alt](public/image.png)

    These devices offer quite different topologies so make sure you pay attention when exploring those. Let's start by defining a simple circuit to run on such hardware devices.
    """)
    return


@app.cell
def _():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2, 2)

    circuit.h(0)
    circuit.cx(0,1)

    circuit.measure([0,1], [0,1])

    circuit.draw('mpl')
    return QuantumCircuit, circuit


@app.cell
def _(circuit):
    from qiskit_aer import AerSimulator
    from qiskit.visualization import plot_histogram

    # execute the quantum circuit
    simulator = AerSimulator()

    result = simulator.run(circuit, shots=1000).result()
    counts  = result.get_counts(circuit)

    plot_histogram(counts)
    return counts, plot_histogram, simulator


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Cool! Now let's see what we get when runing on a real hardware. For that we will first need to connect to IQM service using their library and an authentication TOKEN that allows the notebook to impersonate me.
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
    QUANTUM_COMPUTER = "emerald"

    provider = IQMProvider(IQM_URL, quantum_computer=QUANTUM_COMPUTER)
    backend = provider.get_backend()
    print(f"Connected to : {backend.name}")
    return (backend,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can look for information about the calibration data, errors and such...
    """)
    return


@app.cell
def _(backend):
    import matplotlib.pyplot as plt
    from iqm.qubit_selector.qubit_selector import CalibrationDataManager, CalibrationType

    calibration_data = CalibrationDataManager().get_calibration_fidelities(backend)

    def plot_calibration_data(key, data):
        two_qubit_keys = [CalibrationType.CZ.value, CalibrationType.CLIFFORD.value]
        coherence_keys = [CalibrationType.T1.value, CalibrationType.T2.value]
        every_second = key in two_qubit_keys
        coherence = key in coherence_keys
        xlabel = 'Pairs' if key in two_qubit_keys else 'Qubits'
        ylabel = 'Error' if key not in coherence_keys else 'Coherence in (us)'
        pairs = list(data.items())
        if every_second:
            pairs = pairs[::2]
        labels = [pair[0] for pair in pairs]

        if coherence:
            values = [pair[1] for pair in pairs]
        else:
            values = [1-pair[1] for pair in pairs]

        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{key} metrics')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    for k, v in calibration_data.items():
        plot_calibration_data(k, v)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is some crucial information as it tells us what gates are available to the device and what fidelity or trust we can have when performing our set of operations. Thus, in order to be able to run our circuit we need to translate to the device native set of gates our proposed circuit.
    """)
    return


@app.cell
def _(backend, circuit):
    from qiskit import transpile

    # Propose a set of qubits to place the circuit in
    initial_layout = list(range(circuit.num_qubits))

    circ_iqm = transpile(
        circuit,
        backend=backend,
        initial_layout=initial_layout,
        optimization_level=3,
        seed_transpiler=42
    )

    circ_iqm.draw("mpl", fold=-1)
    return circ_iqm, transpile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see here some of the differences between the proposed theoretical and actual circuit runing on the device.
    """)
    return


@app.cell
def _(backend, circ_iqm):
    hw_job = backend.run(circ_iqm, shots=1000)
    print(f"Standard Job ID: {hw_job.job_id()} with status {hw_job.status()}")
    return (hw_job,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Queued means there might be people submitting jobs to the same device, so we will need to wait until it finishes.
    """)
    return


@app.cell
def _(hw_job):
    counts_hw_std = hw_job.result().get_counts()
    return (counts_hw_std,)


@app.cell
def _(counts, counts_hw_std, plot_histogram):
    plot_histogram([counts, counts_hw_std], legend=["AerSimulator", "IQM Emerald"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Not perfect but there is some evident similarities. The readout errors did create some unwanted probabilities but so far so good. Let's try with a larger example.
    """)
    return


@app.cell
def _(QuantumCircuit, backend, simulator, transpile):
    num_qubits = 7

    large_circ = QuantumCircuit(num_qubits, num_qubits)

    large_circ.h(0)

    for i in range(0, num_qubits-1):
        large_circ.cx(i,i+1)

    large_circ.measure_all(add_bits = False)

    large_results = simulator.run(large_circ, shots=1000).result()
    local_counts  = large_results.get_counts(large_circ)

    large_iqm = transpile(
        large_circ,
        backend=backend,
        initial_layout=list(range(num_qubits)),
        optimization_level=3,
        seed_transpiler=42
    )

    large_hw_job = backend.run(large_iqm, shots=1000)
    print(f"Standard Job ID: {large_hw_job.job_id()} with status {large_hw_job.status()}")
    return large_circ, large_hw_job, large_iqm, local_counts


@app.cell
def _(large_hw_job, local_counts, plot_histogram):
    large_hw_counts = large_hw_job.result().get_counts()

    plot_histogram([local_counts, large_hw_counts], legend=["AerSimulator", "IQM Emerald"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's noisier. All those unwanted bitstrings are part of the noise being added as the depth of the circuit increases.
    """)
    return


@app.cell
def _(circuit):
    circuit.depth()
    return


@app.cell
def _(large_circ):
    large_circ.depth()
    return


@app.cell
def _(large_iqm):
    large_iqm.depth()
    return


if __name__ == "__main__":
    app.run()
