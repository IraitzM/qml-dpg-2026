# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
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
    Running circuits locally emulated works for a while until the circuit size grows and the local resources cannot handle those any more. Moreover, to explore what hardware can do and have a real understanding on what those devices can do for us, you better start exploring the usage of real quantum hardware trough some of the many providers out there.

    # IBM Quantum Platform

    [IBM Quantum Platform](https://quantum.cloud.ibm.com/) is the cloud based access to IBM's superconducting devices. With the free **Open Plan** you get access to several of their Heron processors (156 qubits each, named after cities like `ibm_fez`, `ibm_marrakesh` or `ibm_kingston`).

    The Open Plan comes with 10 minutes of QPU time every 28 days, so we will keep our jobs small and always check things on a simulator first. Let's start by defining a simple circuit to run on such hardware devices.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.image(src="https://github.com/IraitzM/qml-dpg-2026/blob/variant/ibm/Session%202/public/ibm-compute-resources.png?raw=true")
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
    Cool! Now let's see what we get when running on a real hardware. For that we will first need to connect to IBM Quantum Platform using `qiskit-ibm-runtime` and an API key that allows the notebook to impersonate me.

    Hardware queues can be long on the Open Plan. If you want to go through the notebook without waiting (or without spending QPU time), set `USE_FAKE = True`. You will get a local simulator that mimics the topology, native gates and noise of a real IBM device.
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can look for information about the calibration data, errors and such. First, how the qubits are connected and how good each qubit and connection is.
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
    All that information lives in the backend's `target`, so we can also plot it ourselves.
    """)
    return


@app.cell
def _(backend):
    import matplotlib.pyplot as plt

    target = backend.target
    two_qubit_gate = "cz" if "cz" in target.operation_names else "ecr"

    def get_errors(op_name):
        errors = {}
        for qargs, props in target[op_name].items():
            if props is None or props.error is None:
                continue
            # Two qubit gates may be listed in both directions, keep one
            label = "-".join(str(q) for q in sorted(qargs))
            errors.setdefault(label, props.error)
        return errors

    def get_coherence(attribute):
        values = {}
        for q in range(backend.num_qubits):
            value = getattr(backend.qubit_properties(q), attribute, None)
            if value:
                values[str(q)] = value * 1e6
        return values

    calibration_data = {
        "Readout error": get_errors("measure"),
        "SX error": get_errors("sx"),
        f"{two_qubit_gate.upper()} error": get_errors(two_qubit_gate),
        "T1": get_coherence("t1"),
        "T2": get_coherence("t2"),
    }

    def plot_calibration_data(key, data):
        coherence = key in ("T1", "T2")
        xlabel = 'Pairs' if '-' in next(iter(data)) else 'Qubits'
        ylabel = 'Coherence in (us)' if coherence else 'Error'

        plt.figure(figsize=(12, 6))
        plt.bar(list(data.keys()), list(data.values()))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'{key} metrics')
        plt.xticks(rotation=90, fontsize=5)
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
def _(backend):
    print(f"Native gates: {backend.operation_names}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    IBM Heron devices only know about `rz`, `sx`, `x` and `cz` (plus measurements, resets and delays). A circuit that only uses those gates and respects the qubit connectivity is what IBM calls an **ISA circuit** (Instruction Set Architecture), and it is the only kind of circuit the hardware accepts. Qiskit's transpiler, through a _pass manager_, does that translation for us.
    """)
    return


@app.cell
def _(backend, circuit):
    from qiskit.transpiler import generate_preset_pass_manager

    # Propose a set of qubits to place the circuit in
    initial_layout = list(range(circuit.num_qubits))

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        initial_layout=initial_layout,
        optimization_level=3,
        seed_transpiler=42
    )
    isa_circuit = pass_manager.run(circuit)

    isa_circuit.draw("mpl", fold=-1, idle_wires=False)
    return generate_preset_pass_manager, isa_circuit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see here some of the differences between the proposed theoretical and actual circuit running on the device.

    To run it we use the **Sampler** primitive, which returns the measured bitstrings. Note that a single job can hold several circuits, which saves a lot of overhead when you have many of them.
    """)
    return


@app.cell
def _(backend, isa_circuit):
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    sampler = Sampler(mode=backend)

    hw_job = sampler.run([isa_circuit], shots=1000)
    print(f"Job ID: {hw_job.job_id()} with status {hw_job.status()}")
    return hw_job, sampler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Queued means there might be people submitting jobs to the same device, so we will need to wait until it finishes. You can also follow your jobs in the [workloads page](https://quantum.cloud.ibm.com/workloads).
    """)
    return


@app.cell
def _(hw_job):
    # Counts are stored under the name of the classical register, "c" by default
    counts_hw_std = hw_job.result()[0].data.c.get_counts()
    return (counts_hw_std,)


@app.cell
def _(backend, counts, counts_hw_std, plot_histogram):
    plot_histogram([counts, counts_hw_std], legend=["AerSimulator", backend.name])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Not perfect but there is some evident similarities. The readout errors did create some unwanted probabilities but so far so good. Let's try with a larger example.
    """)
    return


@app.cell
def _(
    QuantumCircuit,
    backend,
    generate_preset_pass_manager,
    sampler,
    simulator,
):
    num_qubits = 7

    large_circ = QuantumCircuit(num_qubits, num_qubits)

    large_circ.h(0)

    for i in range(0, num_qubits-1):
        large_circ.cx(i,i+1)

    large_circ.measure_all(add_bits = False)

    large_results = simulator.run(large_circ, shots=1000).result()
    local_counts  = large_results.get_counts(large_circ)

    large_pm = generate_preset_pass_manager(
        backend=backend,
        initial_layout=list(range(num_qubits)),
        optimization_level=3,
        seed_transpiler=42
    )
    large_isa = large_pm.run(large_circ)

    large_hw_job = sampler.run([large_isa], shots=1000)
    print(f"Job ID: {large_hw_job.job_id()} with status {large_hw_job.status()}")
    return large_circ, large_hw_job, large_isa, local_counts


@app.cell
def _(backend, large_hw_job, local_counts, plot_histogram):
    large_hw_counts = large_hw_job.result()[0].data.c.get_counts()

    plot_histogram([local_counts, large_hw_counts], legend=["AerSimulator", backend.name])
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
def _(large_isa):
    large_isa.depth()
    return


if __name__ == "__main__":
    app.run()
