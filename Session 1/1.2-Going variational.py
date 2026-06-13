# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "pennylane==0.45.0",
#     "pylatexenc==2.10",
#     "qiskit==2.4.2",
#     "qiskit-algorithms==0.4.0",
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
    Quantum circuits allow us to encode the dynamics of a system and observe different energy regimes. One common usage is this particular scenario when we would know a dynamic but we would like to obtain the state and energy that minimizes it.

    $$
    H|\psi_{\lambda}\rangle = \lambda_0|\psi_{\lambda}\rangle
    $$
    where $\lambda_0$ is the minimum eigenvalue.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Assuming a simplified version of that problem we could build a circuit such that encodes that dynamic and uses arbitrary values for rotation angles to be defined during execution, allowing for a template-like creation of the circuit and thus making changes on its executions parameters when needed.
    """)
    return


@app.cell
def _():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    a = Parameter('a')

    qc = QuantumCircuit(1)
    qc.ry(a, 0)
    qc.draw()
    return Parameter, QuantumCircuit, a, qc


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a simple rotation over the Y axis of the bloch sphere. So we know that if we measure the observable X of a given state produced by this rotation we get:

    $$
    \langle Y(a) | X | Y(a) \rangle = \sin(\pi a)
    $$

    One way is to explore it analytically but we can also check numerically the results by runing the circuit and also the actual calculation of the $\sin$ function.
    """)
    return


@app.cell
def _(QuantumCircuit, a, qc):
    import numpy as np
    from qiskit.quantum_info import Statevector

    init = QuantumCircuit(1)
    state = Statevector(init)

    val = 0.4
    circ = qc.assign_parameters({a : np.pi*val})
    eval = state.evolve(circ)
    print(eval)
    return Statevector, eval, np, state, val


@app.cell
def _(eval, np, val):
    from qiskit.quantum_info import Pauli

    # Target hamiltonian
    op = Pauli('X')

    print(f"<Ry|X|Ry> : {eval.expectation_value(oper=op)}")
    print(f"Sin function: {np.sin(np.pi * val)}")
    return (op,)


@app.cell
def _(Parameter, QuantumCircuit, Statevector, np, op):
    def check_value(val:float):

        # Initial state |0>
        init = QuantumCircuit(1)
        state = Statevector(init)

        # Parameter
        a = Parameter('a')

        # Ansatz
        qc = QuantumCircuit(1)
        qc.ry(a, 0)

        # Instantiated circuit
        circ = qc.assign_parameters({a : np.pi*val})
        eval = state.evolve(circ)
        print(eval)

        print(f"<Ry|X|Ry> : {eval.expectation_value(oper=op)}")
        print(f"Sin function: {np.sin(np.pi * val)}")

    check_value(0.3)
    return (check_value,)


@app.cell
def _(check_value):
    check_value(0.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One common objetvie when performing a simulation is, for example to find the right set of parameters that produce a given observable outcome.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Derivatives

    In order to fit the ideal set of parameters minimizing a cost or loss function, gradient based approaches are quite popular as being aware of the direction in which this cost function is minimized it may shorten the time required for the method to find it target.

    Quantum circuits have a similar capacity, we can free up the parameters of a given circuit and optimize based on a cost function which in most cases can be represented by the expectation value over some observable. For this, we would need to understand how these derivatives can be computed... at scale.

    Then we know we could calculate its derivative as $\pi\cos(\pi a)$. This may not be an option for when we go to hardware and we make the circuit much more complicated in terms of gates (in particular multiqubit gates). Enter numerical resolution of derivatives!

    We can produce an approximation to our target by leveraging finite differences for numerical approximation, taking into consideration the limit:

    $$
    f^{'}(a) = \lim_{h \rightarrow 0} \frac{f(a + h) - f(a)}{h}
    $$

    which essentially only requires two evaluations of our function.
    """)
    return


@app.cell
def _(a, eval, np, op, qc, state, val):
    h = 0.0001

    # Get the expectation value for f(x)
    exp_val = eval.expectation_value(oper=op)

    # Get the expectation value for f(x+h)
    circ_h = qc.assign_parameters({a : np.pi*(val+h)})
    eval_h = state.evolve(circ_h)
    exp_val_eps = eval_h.expectation_value(oper=op)

    print('Finite difference:', (exp_val_eps - exp_val) / h)
    print('Cosine formula:   ', np.pi * np.cos(np.pi * val))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With a varying number of observables and compositions, this is what automatic differentiation frameworks can provide by tracking values and benefitting from the composable nature of the numerical approximations. More information in [https://www.tensorflow.org/quantum/tutorials/gradients#2_the_need_for_a_differentiator](https://www.tensorflow.org/quantum/tutorials/gradients#2_the_need_for_a_differentiator)
    """)
    return


@app.cell
def _(Parameter, QuantumCircuit, np):
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit import QuantumRegister

    # Instantiate the quantum circuit
    a1 = Parameter('a1')
    a2 = Parameter('a2')

    # Circuit
    q = QuantumRegister(1)
    qcirc = QuantumCircuit(q)
    qcirc.h(q)
    qcirc.rz(a1, q[0])
    qcirc.rx(a2, q[0])

    # Instantiate the Hamiltonian observable 2X+Z
    H = SparsePauliOp.from_list([('X', 2), ('Z',1)])

    # Parameter list
    params = [[np.pi / 4, 0]]

    qcirc.draw('mpl', style = "clifford")
    return H, params, qcirc


@app.cell
def _(H, params, qcirc):
    from qiskit.primitives import StatevectorEstimator
    from qiskit_algorithms.gradients import ParamShiftEstimatorGradient

    # Define the estimator
    estimator = StatevectorEstimator()
    # Define the gradient
    gradient = ParamShiftEstimatorGradient(estimator)

    # Evaluate the gradient of the circuits using parameter shift gradients
    pse_grad_result = gradient.run(qcirc, H,  params).result().gradients
    print('State estimator gradient computed with parameter shift', pse_grad_result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can do the same thing using other frameworks. Frameworks that can handle those derivative caltulations in different ways, perhaps more efficient for when we will perform local simulations and we know we can keep this information in memory...
    """)
    return


@app.cell
def _():
    import pennylane as qml

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RX(weights[2], wires=1)
        return qml.expval(qml.PauliZ(1))

    return circuit, qml


@app.cell
def _(circuit, qml):
    import pennylane.numpy as pnp

    weights = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

    qml.drawer.use_style("pennylane")
    qml.draw_mpl(circuit)(weights)
    return (weights,)


@app.cell
def _(circuit, weights):
    circuit(weights)
    return


@app.cell
def _(circuit, qml, weights):
    qml.gradients.param_shift(circuit)(weights)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Much more convenient and simpler to use... Checkout: https://docs.pennylane.ai/en/stable/introduction/interfaces.html
    """)
    return


if __name__ == "__main__":
    app.run()
