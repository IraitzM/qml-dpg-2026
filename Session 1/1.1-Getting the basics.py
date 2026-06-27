# /// script
# dependencies = [
#     "ipython==9.14.1",
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "pylatexenc==2.10",
#     "qiskit==2.4.2",
#     "qiskit-aer==0.17.2",
#     "seaborn==0.13.2",
#     "sympy==1.14.0",
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
    # Simulations

    Now, do we have to create our own set of gates and operations? Well, it might be a good practice as the formalism can be easily reproduced by setting the basic needs:

    * Basis $|0\rangle$ and $|1\rangle$ states.
    * Tensor product operation
    * A universal set of gates ($R_x(\theta), R_x(\theta), R_x(\theta), P(\phi)$ phase shift gate and $CNOT$ may suffice).

    We can then create our own set of functions and objects to simulate those computations.

    ## From scratch

    We would easily create basic vector structures for our _quantum framework_. The minimum unit is the qubit and in order to frame the potential quantum states it may hold we would need to create the computational basis set $\{|0\rangle, |1\rangle \}$.
    """)
    return


@app.cell
def _():
    import numpy as np
    from qiskit.visualization import array_to_latex

    zero = [[1], [0]]

    array_to_latex(array=zero, prefix='|0\\rangle = ', max_size=(10,10))
    return array_to_latex, np, zero


@app.cell
def _(array_to_latex):
    one = [[0], [1]]

    array_to_latex(array=one, prefix='|1\\rangle = ', max_size=(10,10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now lets try with some gates.

    $$
    X = \left[
    \begin{array}{cc}
    0 & 1 \\
    1 & 0
    \end{array}
    \right] \quad
    H = \frac{1}{\sqrt{2}}\left[
    \begin{array}{cc}
    1 & 1 \\
    1 & -1
    \end{array}
    \right]
    $$
    """)
    return


@app.cell
def _(array_to_latex):
    X = [[0,1],[1,0]]

    array_to_latex(array=X, prefix='X = ', max_size=(10,10))
    return (X,)


@app.cell
def _(array_to_latex, np):
    hadamard = np.dot((1/(np.sqrt(2))), [[1, 1], [1, -1]])

    array_to_latex(array=hadamard, prefix='H = ', max_size=(10,10))
    return (hadamard,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Well, it is already taking shape. We can test if the outcome matches our expectations.
    """)
    return


@app.cell
def _(array_to_latex, hadamard, np, zero):
    superposition = np.dot(hadamard, zero)

    array_to_latex(array=superposition, prefix='H|0\\rangle = |+\\rangle = ', max_size=(10,10))
    return (superposition,)


@app.cell
def _(superposition):
    superposition
    return


@app.cell
def _(X, array_to_latex, np, zero):
    one_sim = np.dot(X, zero)

    array_to_latex(array=one_sim, prefix='X|0\\rangle = |1\\rangle = ', max_size=(10,10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can scale it to a couple of qubits to see what we get. Let's try to create on of the bell states we saw during class.

    $$
    |\Phi^{+}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \quad |\Phi^{-}\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)
    $$
    $$
    |\Psi^{+}\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) \quad |\Psi^{-}\rangle = \frac{1}{\sqrt{2}}(|10\rangle - |10\rangle)
    $$

    We will need the CNOT gate for this.
    """)
    return


@app.cell
def _(array_to_latex):
    CNOT = [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]

    array_to_latex(array=CNOT, prefix='CNOT = ', max_size=(10,10))
    return (CNOT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With this we will create a two qubit system, apply the Hadamard gate to the first one and the CNOT gate with the control over the first qubit as well.
    """)
    return


@app.cell
def _(hadamard, np, zero):
    # Initial state
    init_state = np.kron(zero, zero)

    # (I(2) tensor Hadamard)
    HI = np.kron(hadamard, np.eye(2))
    return HI, init_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With that we can perform the full operation

    $$
    CNOT (I\otimes H)|00\rangle = |\Phi^{+}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle
    $$
    """)
    return


@app.cell
def _(CNOT, HI, array_to_latex, init_state, np):
    psi_1 = np.dot(HI, init_state)
    psi = np.dot(CNOT, psi_1)

    array_to_latex(array=psi, prefix='|\\psi\\rangle = ', max_size=(10,10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Well, having a theoretical framework may be a good option for simulating and doing some local experimentation but that will reach soon limitations when trying to scale it up. Our classical device won't be able to perform the whole system calculations and we might need to switch to actual quantum computers doing those. Therefore, we need to find a way to o so.

    This is why some manufacturers have invested time and effort on creating open-source frameworks to be adopted by the community (and position themselves). Companies such as [IBM](https://qiskit.org/) or [AWS](https://github.com/aws/amazon-braket-sdk-python) have leveraged their own version that also allows for these programs to be sent to an end device that will perform the set of operations.
    """)
    return


@app.cell(hide_code=True)
def _():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(2, 2)
    qc.x(0)
    qc.cx(0, 1)
    qc.measure([0,1], [0,1])

    qc.draw('mpl')
    return (QuantumCircuit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Qiskit

    Qiskit is IBM's quantum computing toolkit the enables interaction with their devices. Let's first start replicating the theoretical basis and then move forward up to device simulation.

    Opflow is the framework section dedicated to provide the pieces to perform previous computations.
    """)
    return


@app.cell
def _():
    from qiskit.quantum_info import Statevector

    Zero = Statevector.from_label('0')
    One = Statevector.from_label('1')
    return One, Statevector, Zero


@app.cell
def _(Zero):
    probs = Zero.probabilities()
    print('Probability of measuring 0: {}'.format(probs[0]))
    return (probs,)


@app.cell
def _(probs):
    print('Probability of measuring 1: {}'.format(probs[1]))
    return


@app.cell
def _(Statevector):
    Plus = Statevector.from_label('+')

    print(Plus)
    return (Plus,)


@app.cell
def _(Plus):
    Plus.probabilities()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Qiskit tends to understand everything in terms of circuits. But in essence we can request the actual operation being performed there and check that the Hadamard action over a $|0\rangle$ state (the initial state) provides the $|+\rangle$ state as expected.
    """)
    return


@app.cell
def _(Plus, array_to_latex):
    array_to_latex(array=Plus.data, prefix='|+\\rangle = ', max_size=(10,10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What would be the outcome of it?
    """)
    return


@app.cell
def _(Plus):
    probs_2 = Plus.probabilities()
    print('Probability of measuring 0: {}'.format(probs_2[0]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is the expected outcome for the measurement operation ($|\langle 0 | \psi \rangle|^2$). But we can mimic it as well.
    """)
    return


@app.cell
def _(Plus, Zero, np):
    print(abs(np.dot(Zero.data, Plus.data.T)**2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly gate operations can be used by simply calling the name of it.
    """)
    return


@app.cell
def _():
    from qiskit.quantum_info import Pauli

    Xgate = Pauli('X')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And what is the amplitude of $|1\rangle$ after the $X|0\rangle$ operation?
    """)
    return


@app.cell
def _(One, X, Zero):
    Zero.evolve(X) == One
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Qiskit orders bits in a specific manner so some gates may look different but is just a matter of ordering when applying the operations.
    """)
    return


@app.cell
def _():
    from qiskit.circuit.library import UnitaryGate

    matrix = [[1., 0., 0., 0.],
              [0., 0., 0., 1.],
              [0., 0., 1., 0.],
              [0., 1., 0., 0.]]
    CNOTgate = UnitaryGate(matrix)
    return (CNOTgate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let play around with the Bell state we produced before

    $$
    CNOT (I\otimes H)|00\rangle = |\Phi^{+}\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle
    $$
    """)
    return


@app.cell
def _(Zero):
    print(Zero ^ Zero)
    return


@app.cell
def _(QuantumCircuit):
    qc_bell = QuantumCircuit(2)
    qc_bell.h(0)

    qc_bell.draw()
    return (qc_bell,)


@app.cell
def _(qc_bell):
    from qiskit.quantum_info import Operator

    print(Operator(qc_bell).data)
    return


@app.cell
def _(Zero, qc_bell):
    (Zero ^ Zero).evolve(qc_bell).data
    return


@app.cell
def _(CNOTgate, Zero, qc_bell):
    bell_state = (Zero ^ Zero).evolve(qc_bell).evolve(CNOTgate)
    bell_state.data
    return (bell_state,)


@app.cell
def _(array_to_latex, bell_state):
    array_to_latex(array=bell_state.data, prefix='|\\psi\\rangle = ', max_size=(10,10))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Of course the gate-based nature of IBM devices makes it more natural to directly code our approach as a gate based circuit.
    """)
    return


@app.cell
def _(QuantumCircuit):
    circ = QuantumCircuit(2)
    circ.h(0)
    circ.cx(0,1)

    circ.draw('mpl')
    return (circ,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Most likelly moving forward, this pictorical approach will ease the abstraction but is good to know that, the formalism is still there.
    """)
    return


@app.cell
def _(Zero, circ):
    print('Math:', (Zero ^ Zero).evolve(circ).probabilities())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Of course qiskit offers some other nice ways to simulate and visualize the results.
    """)
    return


@app.cell
def _(Statevector, circ):
    from qiskit.visualization import plot_state_qsphere

    psi_state  = Statevector.from_instruction(circ)

    plot_state_qsphere(psi_state)
    return plot_state_qsphere, psi_state


@app.cell
def _(psi_state):
    from qiskit.visualization import plot_bloch_multivector

    plot_bloch_multivector(psi_state)
    return (plot_bloch_multivector,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to simulate the actual action of the circuit we will need to add some classical registers and measurement operations.
    """)
    return


@app.cell
def _(QuantumCircuit, circ):
    circuit = QuantumCircuit(2, 2)

    circuit = circuit.compose(circ)

    circuit.measure([0,1], [0,1])

    circuit.draw('mpl')
    return (circuit,)


@app.cell
def _(circuit):
    from qiskit_aer import AerSimulator

    # execute the quantum circuit
    simulator = AerSimulator()

    result = simulator.run(circuit, shots=1000).result()
    counts  = result.get_counts(circuit)
    print(counts)
    return counts, simulator


@app.cell
def _(counts):
    from qiskit.visualization import plot_histogram

    plot_histogram(counts)
    return (plot_histogram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise

    Could we create a state that superposes all potential basis states for a 2-qubit configuration?

    $$
    \frac{1}{2}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|10\rangle + \frac{1}{2}|11\rangle
    $$
    """)
    return


@app.cell
def _(QuantumCircuit):
    exec_qc = QuantumCircuit(2)

    # HERE GOES YOUR CIRCUIT
    return (exec_qc,)


@app.cell
def _(Statevector, exec_qc, plot_bloch_multivector):
    psi_exec  = Statevector.from_instruction(exec_qc)

    plot_bloch_multivector(psi_exec)

    return (psi_exec,)


@app.cell
def _(plot_state_qsphere, psi_exec):
    plot_state_qsphere(psi_exec)
    return


@app.cell
def _(QuantumCircuit, circuit, exec_qc, plot_histogram, simulator):
    exec_circ = QuantumCircuit(2, 2)
    exec_circ = circuit.compose(exec_qc)
    exec_circ.measure([0,1],[0,1])

    exec_result = simulator.run(exec_circ, shots=1000).result()
    exec_counts  = exec_result.get_counts(exec_circ)

    plot_histogram(exec_counts)
    return


if __name__ == "__main__":
    app.run()
