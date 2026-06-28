# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "networkx==3.6.1",
#     "numpy==2.4.6",
#     "pennylane==0.45.0",
#     "quimb==1.14.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Tensors

    Tensors are multi-dimensional arrays of numbers. Intuitively, they can be interpreted as a generalization of scalars, vectors, and matrices. Tensors can be described by their rank, indices and the dimension of the indices. The rank is the number of indices in a tensor — a scalar has rank zero, a vector has rank one, and a matrix has rank two. The dimension of an index is the number of values that index can take. For example, a vector with three elements has one index that can take three values. This is vector is therefore a rank one tensor and its index has dimension three.
    """)
    return


@app.cell
def _():
    import numpy as np

    tensor_rank1 = np.array([1, 2, 3, 4])
    print("rank: ", tensor_rank1.ndim)
    print("dimensions: ", tensor_rank1.shape)
    return np, tensor_rank1


@app.cell
def _(np, tensor_rank1):
    tensor_rank2 = np.array([tensor_rank1, tensor_rank1, tensor_rank1])
    print("rank: ", tensor_rank2.ndim)
    print("dimensions: ", tensor_rank2.shape)
    return (tensor_rank2,)


@app.cell
def _(np, tensor_rank2):
    tensor_rank3 = np.array([tensor_rank2, tensor_rank2])
    print("rank: ", tensor_rank3.ndim)
    print("dimensions: ", tensor_rank3.shape)
    print("Rank-3 tensor: \n", tensor_rank3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Picturing graphically this tensors will help us down the road on some concepts related to the operations we will perform with those. Let's take a tensor describing a quantum state $|\psi\rangle$.

    Let's make this state the $|\Psi^{-}\rangle$ bell state

    $$
    |\Psi^{-}\rangle = \frac{1}{\sqrt{2}}(|0\rangle_A \otimes|1\rangle_B - |1\rangle_A \otimes|0\rangle_B)
    $$
    """)
    return


@app.cell
def _():
    import quimb as qu
    import quimb.tensor as qtn

    data = qu.bell_state("psi-").reshape(2, 2)
    inds = ("A", "B")
    tags = ("KET",)

    ket = qtn.Tensor(data=data, inds=inds, tags=tags)
    ket
    return ket, qtn, qu


@app.cell
def _(ket):
    ket.data.reshape(4,-1)
    return


@app.cell
def _(ket):
    ket.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could express operations taking place at different indices.
    """)
    return


@app.cell
def _(qtn, qu):
    X = qtn.Tensor(qu.pauli("X"), inds=("A", "C"), tags=["PAULI", "X", "0"])

    print(X.data)
    return (X,)


@app.cell
def _(qtn, qu):
    Y = qtn.Tensor(qu.pauli("Y"), inds=("B", "D"), tags=["PAULI", "Y", "1"])

    print(Y.data)
    return (Y,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And connect this two indices to a state we would like to project to.

    $$
    \langle \Psi^{-}|YX|\Psi^{-}\rangle
    $$
    """)
    return


@app.cell
def _(qtn, qu):
    bra = qtn.Tensor(qu.bell_state("psi-").reshape(2, 2), inds=("C", "D"), tags=["BRA"])
    return (bra,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus, by virtue of mathematical formulation our result can be defined as
    """)
    return


@app.cell
def _(X, Y, bra, ket):
    TN = ket.H & X & Y & bra
    TN
    return (TN,)


@app.cell
def _(TN):
    TN.draw()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Quimb](https://quimb.readthedocs.io/en/latest/) allows to contract some of the tags so we can see it closer to our analytical formula.
    """)
    return


@app.cell
def _(TN):
    TNc = TN ^ "PAULI"
    TNc.draw("PAULI")
    return (TNc,)


@app.cell
def _(TNc):
    print(TNc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This formulation allows for some operations of interest for us quantum computing enthusiasts.
    """)
    return


@app.cell
def _(TNc):
    TNc.contract()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Is that right?

    **Step 1: Apply X to qubit 2**

    $$X|0\rangle = |1\rangle, \quad X|1\rangle = |0\rangle$$

    $$X \otimes I \text{ on the second qubit gives:}$$

    $$(Y \otimes X)|\Psi^-\rangle = \frac{1}{\sqrt{2}}\left(Y|0\rangle \otimes X|1\rangle - Y|1\rangle \otimes X|0\rangle\right)$$

    $$= \frac{1}{\sqrt{2}}\left(Y|0\rangle \otimes |0\rangle - Y|1\rangle \otimes |1\rangle\right)$$

    **Step 2: Apply Y to qubit 1**

    $$Y|0\rangle = i|1\rangle, \quad Y|1\rangle = -i|0\rangle$$

    $$= \frac{1}{\sqrt{2}}\left(i|1\rangle|0\rangle - (-i)|0\rangle|1\rangle\right) = \frac{i}{\sqrt{2}}\left(|10\rangle + |00\rangle... \right)$$

    Wait, let me be careful:

    $$= \frac{1}{\sqrt{2}}\left(i|10\rangle + i|01\rangle\right) = \frac{i}{\sqrt{2}}\left(|10\rangle + |01\rangle\right)$$

    **Step 3: Take the inner product with $\langle\Psi^-|$**

    $$\langle\Psi^-| = \frac{1}{\sqrt{2}}\left(\langle 01| - \langle 10|\right)$$

    $$\langle\Psi^-|(Y\otimes X)|\Psi^-\rangle = \frac{1}{\sqrt{2}}\left(\langle 01| - \langle 10|\right) \cdot \frac{i}{\sqrt{2}}\left(|10\rangle + |01\rangle\right)$$

    $$= \frac{i}{2}\left(\langle 01|01\rangle + \langle 01|10\rangle - \langle 10|10\rangle - \langle 10|01\rangle\right)$$

    $$= \frac{i}{2}\left(1 + 0 - 1 - 0\right) = \frac{i}{2}(0) = 0$$

    ---

    $$\boxed{\langle\Psi^-|YX|\Psi^-\rangle = 0}$$

    > **Why is it zero?** The operator $Y \otimes X$ is **Hermitian** with eigenvalues $\pm 1$. However, the singlet state $|\Psi^-\rangle$ has a special antisymmetry — it gives zero for many tensor products of the form $A \otimes B$ when the combination is "wrong" relative to its symmetry. Specifically, the singlet is invariant under $U \otimes U^*$ for any unitary $U$, and this symmetry forces many expectation values to vanish. You get nonzero results for operators like $X \otimes X$, $Y \otimes Y$, $Z \otimes Z$ (each gives $-1$), but cross-combinations like $Y \otimes X$ yield $0$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tensor contraction and decomposition

    Matrix-matrix and matrix-vector multiplications are familiar operations within the context of quantum computing. We can now study these operations through the lens of the tensor notation introduced above. To define tensor networks, it is important to first understand tensor contraction. Two or more tensors can be contracted by summing over repeated indices. In diagrammatic notation, the repeated indices appear as lines connecting tensors, as in the figure below. We see two tensors of rank two connected by one repeated index, The dimension of the repeated index is called the bond dimension.

    The contraction of the tensors in the above example is equivalent to the standard matrix multiplication formula and can be expressed as

    $$
    C_{ij} = \sum_k A_{ik}B_{kj}
    $$

    where $C_{ij}$ denotes the entry for the $i$-th row and $j$-th column of the product $C = AB$.

    Using the tensor product $\otimes$ between the 3 tensors and summing over the repeated indices we can obtain a similar expression for the full tensor [2](https://iopscience.iop.org/article/10.1088/1751-8121/aa6dc3).

    $$
    D_{iln} = \sum_{jkm}A_{ijk}\otimes B_{jlm} \otimes C_{kmn}
    $$

    ![ex](https://blog-assets.cloud.pennylane.ai/demos/tutorial_tensor_network_basics/main/_assets/images/06-tensor-tensor.png?w=828)
    """)
    return


@app.cell
def _(qtn):
    ta = qtn.rand_tensor([2, 3], inds=["a", "x"], tags="A")
    tb = qtn.rand_tensor([4, 3], inds=["b", "x"], tags="B")

    # matrix multiplication but with indices aligned automatically
    ta @ tb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Matrix Product States

    An MPS represents a many-body quantum state as a chain of tensors — one per site. Instead of storing all $2^N$ amplitudes of an $N$-qubit state, you store $N$ small tensors connected by shared "bond" indices. The full state is recovered by contracting the chain.

    Formally, for a system of $N$ sites each with local dimension $d$ (e.g. $d=2$ for qubits):

    $$|\psi\rangle = \sum_{\sigma_1, \ldots, \sigma_N} A^{\sigma_1}[1] \cdot A^{\sigma_2}[2] \cdots A^{\sigma_N}[N] \; |\sigma_1 \sigma_2 \cdots \sigma_N\rangle$$

    Each $A^{\sigma_i}[i]$ is a matrix (or rank-3 tensor with indices: left-bond, physical, right-bond). The chain of matrix multiplications gives a scalar for each basis state.

    **Bond dimension** $\chi$ is the size of the shared index between adjacent tensors. It controls how much entanglement the MPS can represent. A bond of dimension $\chi$ means the matrices are $\chi \times \chi$ — so the total memory scales as $O(N d \chi^2)$ instead of $O(d^N)$.
    """)
    return


@app.cell
def _(qtn):
    # Random MPS: N sites, local dim d=2, bond dim chi
    psi = qtn.MPS_rand_state(L=6, bond_dim=4, phys_dim=2)

    # Access individual site tensors
    print("left boundary: d × χ ", psi[0].shape)   # (2, 4)
    print("bulk: χ × d × χ ", psi[1].shape)   # (4, 2, 4)
    print("right boundary: χ × d ", psi[5].shape)   # (4, 2)

    # Bond dimensions
    print("Bond dimensions ", psi.bond_sizes())   # [4, 4, 4, 4, 4]
    return (psi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Product state (no entanglement)
    """)
    return


@app.cell
def _(qtn):
    psi_prod = qtn.MPS_rand_state(L=6, bond_dim=1, phys_dim=2)
    psi_prod
    return (psi_prod,)


@app.cell
def _(psi_prod):
    len(psi_prod.contract().data.reshape(-1)) == 2**6
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Week entanglement
    """)
    return


@app.cell
def _(qtn):
    qtn.MPS_rand_state(L=6, bond_dim=2, phys_dim=2)
    return


@app.cell
def _(psi):
    # compute its norm squared
    psi.H @ psi  # == (tn.H & tn) ^ ...
    return


@app.cell
def _(qtn):
    # create a tensor with 5 legs
    t = qtn.rand_tensor([2, 3, 4, 5, 6], inds=["a", "b", "c", "d", "e"])
    t.draw(figsize=(3, 3))
    return (t,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we were provided with a definition like the above we could split it into groups based on indices to make it easier to manage.
    """)
    return


@app.cell
def _(t):
    # split the tensor, by grouping some indices as 'left'
    tn = t.split(["a", "c", "d"])
    tn.draw(figsize=(3, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus a Tensor Network is as its name states a set of tensors knitted based on indices and bond dimensions that in the case of quantum computer account for how much entanglement it can handle.
    """)
    return


@app.cell
def _(qtn):
    circ = qtn.TensorNetwork(
        (
            qtn.rand_tensor([2, 3, 3], ["a", "b", "c"], "T0"),
            qtn.rand_tensor([3, 3, 3, 4], ["c", "d", "e", "f"], "T1"),
            qtn.rand_tensor([3, 3, 4], ["d", "g", "h"], "T2"),
            qtn.rand_tensor([4, 3, 4], ["h", "b", "f"], "T3"),
        )
    )
    circ.draw(["T0", "T1", "T2", "T3"], show_inds="all")
    return (circ,)


@app.cell
def _(circ):
    print("outer inds:", circ.outer_inds())
    return


@app.cell
def _(circ):
    circ.contract().draw(["T0", "T1", "T2", "T3"], figsize=(4, 4))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From TN to QC

    A tensor network is a collection of tensors where a subset of all indices are contracted. As mentioned above, we can use diagrammatic notation to specify which indices and tensors will be contracted together by connecting individual tensors with lines. Tensor networks can represent complicated operations involving several tensors with many indices contracted in sophisticated patterns.

    Two well-known tensor network architectures are matrix product states ([MPS](https://docs.pennylane.ai/en/stable/code/api/pennylane.MPS.html)) and tree tensor networks ([TTN](https://docs.pennylane.ai/en/stable/code/api/pennylane.TTN.html)). These follow specific patterns of connections between tensors and can be extended to have many or few indices. Examples of these architectures with only a few tensors can be seen in the figure below. An MPS is shown on the left and a TTN on the right.

    ![tn2](https://blog-assets.cloud.pennylane.ai/demos/tutorial_tn_circuits/main/_assets/images/MPS_TTN_Color.PNG?w=828)

    These tensor networks are commonly used to efficiently represent certain many-body quantum states [[1](https://www.sciencedirect.com/science/article/abs/pii/S0003491614001596)]. Every quantum circuit can be represented as a tensor network, with the bond dimension dependent on the width and connectivity of the circuit. Moreover, one can design quantum circuits that have the same connectivity as well-known tensor networks like MPS and TTN. We call these tensor-network quantum circuits. Note that the connectivity of a tensor network is related to how entanglement is distributed and how correlations spread in the resulting tensor-network quantum circuit. We therefore design circuits based on the tensor networks that best capture the information we want to extract.

    In tensor-network quantum circuits, the tensor network architecture acts as a guideline for the shape of the quantum circuit. More specifically, the tensors in the tensor networks above are replaced with unitary operations to obtain quantum circuits, as illustrated in the figure below.

    ![mps](https://blog-assets.cloud.pennylane.ai/demos/tutorial_tn_circuits/main/_assets/images/MPS_TTN_Circuit_Color.PNG?w=828)

    Since the unitary operations $U_1$ to $U_3$ are in principle completely general, it is not always clear how to implement them with a specific gate set. Instead, we can replace the unitary operations with variational quantum circuits determined by a specific template of choice. The PennyLane tensor network templates allow us to do precisely this: implement tensor-network quantum circuits with user-defined circuit ansatze as the unitary operations (ex. [MERA](https://docs.pennylane.ai/en/stable/code/api/pennylane.MERA.html)). In this sense, just as a template is a strategy for arranging parametrized gates, tensor-network quantum circuits are strategies for structuring circuit templates. They can therefore be interpreted as templates of templates, i.e., as meta-templates.

    ![circuit](https://blog-assets.cloud.pennylane.ai/demos/tutorial_tensor_network_basics/main/_assets/images/11-circuit.png?w=828)

    [Source](https://pennylane.ai/demos/tutorial_tn_circuits)
    """)
    return


@app.cell
def _():
    import numpy as onp
    import pennylane as qp

    # MPS block
    def block(weights, wires):
        qp.RX(weights[0], wires=wires[0])
        qp.RY(weights[1], wires=wires[1])
        qp.CNOT(wires=wires)

    # Default
    dev = qp.device("default.qubit", wires=4)

    @qp.qnode(dev)
    def circuit(template_weights):
        # Matrix Product State as QC
        qp.MPS(
            wires=range(4),
            n_block_wires=2,
            block=block,
            n_params_block=2,
            template_weights=template_weights,
        )
        return qp.expval(qp.PauliZ(wires=3))

    return circuit, qp


@app.cell
def _(circuit, np, qp):
    import matplotlib.pyplot as plt

    np.random.seed(1)
    weights = np.random.random(size=[3, 2])
    qp.drawer.use_style("black_white")
    fig, ax = qp.draw_mpl(circuit, level="device")(weights)
    fig.set_size_inches((6, 3))
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Simulating circuits just by using the right mathematical [backend](https://pennylane.ai/demos/tutorial_How_to_simulate_quantum_circuits_with_tensor_networks).
    """)
    return


@app.cell
def _(np, qp):
    # Define the keyword arguments for the MPS method
    kwargs_mps = {
        # Maximum bond dimension of the MPS
        "max_bond_dim": 50,
        # Cutoff parameter for the singular value decomposition
        "cutoff": np.finfo(np.complex128).eps,
        # Contraction strategy to apply gates
        "contract": "auto-mps",
    }

    # Parameters of the quantum circuit
    theta = 0.5
    phi = 0.1

    # Instantiate the device with the MPS method and the specified kwargs
    tn_dev = qp.device("default.tensor", method="mps", **kwargs_mps)

    # Define the quantum circuit
    @qp.qnode(tn_dev)
    def tn_circ(theta, phi, num_qubits):
        for qubit in range(num_qubits - 4):
            qp.RX(theta, wires=qubit + 1)
            qp.CNOT(wires=[qubit, qubit + 1])
            qp.RY(phi, wires=qubit + 1)
            qp.DoubleExcitation(theta, wires=[qubit, qubit + 1, qubit + 3, qubit + 4])
            qp.Toffoli(wires=[qubit + 1, qubit + 3, qubit + 4])
        return qp.expval(
            qp.X(num_qubits - 1) @ qp.Y(num_qubits - 2) @ qp.Z(num_qubits - 3)
        )

    return phi, theta, tn_circ


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We set the maximum bond dimension to 50 and the cutoff parameter is set to the machine epsilon of the numpy.complex128 data type. For this circuit, retaining a maximum of 50 singular values in the singular value decomposition is more than enough to represent the quantum state accurately. Finally, the contraction strategy is set to auto-mps. For an explanation of these parameters, we refer to the documentation of the default.tensor device.

    As a general rule, choosing the appropriate method and setting the optimal keyword arguments is essential to achieve the best performance for a given quantum circuit. However, the optimal choice depends on the specific circuit structure. For optimization tips, we also refer to the performance checklist in the quimb documentation.

    We can now simulate the quantum circuit for different numbers of qubits. The execution time will generally increase as the number of qubits grows. The first execution is typically slower due to the initial setup and compilation processes of quimb.
    """)
    return


@app.cell
def _(phi, theta, tn_circ):
    import time

    # Simulate the circuit for different numbers of qubits
    for num_qubits in range(50, 201, 50):
        print(f"Number of qubits: {num_qubits}")
        start_time = time.time()
        result = tn_circ(theta, phi, num_qubits)
        end_time = time.time()
        print(f"Result: {result}")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Wow, 200 qubit system is hard to simulate if you try a traditional simulator... this is part of the magic when it comes to QC simulation through tensor networks and the reason behind all the papers related to tensor networks trashing quantum advantage claims...

    * [Quantum Advantage: A Tensor Network Perspective](https://arxiv.org/abs/2603.18825)

    Once defined as gates sequences, Quantum Circuits can be also converted to their TN representation as [Matrix Product Operator (MPO)](https://quimb.readthedocs.io/en/latest/examples/ex_circuit_to_mpo.html) but often times, if we rely on a provider like Pennylane this will be transparent to us, simply benefiting from the performance under the hood.

    Some good examples on how to use this framework for quantum computing practitioners can be found in:

    * Pennylane's [documentation](https://pennylane.ai/demos/tutorial_tensor_network_basics)
    * Quimb library [documentation](https://quimb.readthedocs.io/en/latest/tensor/tensor-basics.html)
    """)
    return


if __name__ == "__main__":
    app.run()
