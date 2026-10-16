# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "networkx==3.6.1",
#     "numpy==2.4.6",
#     "pygraphviz==2.0.1",
#     "quimb==1.14.0",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise: contracting tensors, and paying for entanglement

    A hands-on companion to `3.1-Tensor networks.py`.

    ## Exercises

    1. **Contract** two tensors over a shared index by hand, with plain
       `numpy`, and check it against `quimb`'s automatic index alignment
       (the `ta @ tb` trick from "Contraction" section).
    2. **Count** the parameters an MPS actually stores, as a formula, and
       use it to see why bond dimension $\chi$, not qubit count, is what
       you pay for.
    3. **Decompose** a single-qubit unitary tensor into Pauli matrices, the
       first step of the reference notebook's §"From TN to QC" recipe for
       turning a tensor-network diagram into an actual quantum circuit.
    4. **Simulate** a small circuit as an MPS and read off a measurement
       probability, the payoff for everything above: contraction,
       storage cost, and gate decomposition all feed into actually
       running a circuit.

    Same shape as the other `Extra/` exercises: a stub function to
    complete, a **✅ Check** button, and a **💡 Show solution** button,
    collapsed by default.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import quimb as qu
    import quimb.tensor as qtn

    return np, plt, qtn, qu


@app.cell
def _(np):
    PAULI = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    return (PAULI,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: Contract two tensors over a shared index

    The reference notebook builds two rank-2 tensors that share an index
    named `"x"` but disagree on *where* that index sits:

    ```python
    ta = qtn.rand_tensor([2, 3], inds=["a", "x"], tags="A")
    tb = qtn.rand_tensor([4, 3], inds=["b", "x"], tags="B")
    ta @ tb
    ```

    `quimb` contracts by matching index **names**, not axis positions, so
    `ta @ tb` sums over `"x"` (axis 1 of both arrays) and returns a tensor
    indexed by `("a", "b")`, an ordinary matrix multiplication once you
    account for the fact that neither operand needs transposing first.
    In formula form, for $A$ shaped $(m, k)$ and $B$ shaped $(n, k)$,
    sharing their *second* axis:

    $$C_{ij} = \sum_k A_{ik} B_{jk}$$

    **Implement `contract_shared_index(A, B)`** for two 2D numpy arrays
    `A` (shape `(m, k)`) and `B` (shape `(n, k)`), returning the `(m, n)`
    array `C` above. `np.einsum` is the natural tool here, though a
    double loop with explicit sums works too.
    """)
    return


@app.function
def contract_shared_index(A, B):
    """C[i, j] = sum_k A[i, k] * B[j, k], for A: (m, k) and B: (n, k).

    TODO: see the exercise text above.
    """
    raise NotImplementedError(
        "Exercise 1: implement contract_shared_index(A, B)"
    )


@app.cell(hide_code=True)
def _(mo):
    ex1_check = mo.ui.run_button(label="✅ Check Exercise 1")
    ex1_check
    return (ex1_check,)


@app.cell(hide_code=True)
def _(ex1_check, mo, np, qtn):
    mo.stop(
        not ex1_check.value,
        mo.md("*Fill in `contract_shared_index` above, then click the button.*"),
    )

    def _run():
        rng = np.random.default_rng(0)
        rows = ["| A shape | B shape | matches quimb's `ta @ tb` |", "|---|---|---|"]
        all_ok = True
        for m, k, n in [(2, 3, 4), (5, 2, 2), (3, 3, 3)]:
            A = rng.normal(size=(m, k))
            B = rng.normal(size=(n, k))
            ta = qtn.Tensor(A, inds=("a", "x"))
            tb = qtn.Tensor(B, inds=("b", "x"))
            want = (ta @ tb).data
            got = contract_shared_index(A, B)
            ok = got.shape == want.shape and np.allclose(got, want)
            all_ok = all_ok and ok
            rows.append(f"| {A.shape} | {B.shape} | {'✅' if ok else '❌'} |")
        verdict = "**All checks passed ✅**" if all_ok else "**Something's off ❌**"
        return mo.vstack([mo.md("\n".join(rows)), mo.md(verdict)])

    try:
        _result = _run()
    except NotImplementedError as _e:
        _result = mo.md(f"*Not implemented yet: {_e}*")
    _result
    return


@app.cell(hide_code=True)
def _(mo):
    ex1_solution = mo.ui.run_button(label="💡 Show solution 1")
    ex1_solution
    return (ex1_solution,)


@app.cell(hide_code=True)
def _(ex1_solution, mo):
    mo.stop(not ex1_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def contract_shared_index(A, B):
        return np.einsum("ik,jk->ij", A, B)
    ```

    Notice this is **not** `A @ B` (ordinary matrix multiplication), which
    would need `B` transposed first since matmul always contracts the
    *last* axis of the left operand against the *first* axis of the
    right one. Tensor contraction has no such positional rule, it
    contracts whichever axes share a name, wherever they happen to sit.
    That's the entire reason `quimb` asks for `inds` on every tensor: the
    index names *are* the wiring diagram, position on the array is just
    bookkeeping.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: How many numbers does an MPS actually store?

    An open-boundary MPS on $N$ sites, local dimension $d$, uniform bond
    dimension $\chi$, is a chain of $N$ tensors:

    - the **two boundary tensors** (first and last site) each have one
      bond index and one physical index: $d \times \chi$ numbers apiece;
    - the **$N-2$ bulk tensors** each have two bond indices and one
      physical index: $d \times \chi^2$ numbers apiece.

    $$\text{params}(N, d, \chi) = \underbrace{2\,d\,\chi}_{\text{2 boundary tensors}} \;+\; \underbrace{(N-2)\,d\,\chi^2}_{\text{$N-2$ bulk tensors}}$$

    (For $N=2$ there are no bulk tensors and the second term is $0$,
    both tensors are boundary tensors sharing the single bond between
    them.)

    **Implement `mps_param_count(N, d, chi)`** as exactly that formula,
    for $N \geq 2$.
    """)
    return


@app.function
def mps_param_count(N, d, chi):
    """Total number of entries across all N tensors of an open-boundary
    MPS with local dimension d and uniform bond dimension chi.

    TODO: see the exercise text above.
    """
    raise NotImplementedError("Exercise 2: implement mps_param_count(N, d, chi)")


@app.cell(hide_code=True)
def _(mo):
    ex2_check = mo.ui.run_button(label="✅ Check Exercise 2")
    ex2_check
    return (ex2_check,)


@app.cell(hide_code=True)
def _(ex2_check, mo, qtn):
    mo.stop(
        not ex2_check.value,
        mo.md("*Fill in `mps_param_count` above, then click the button.*"),
    )

    def _run():
        rows = [
            "| N | d | χ | your formula | quimb's actual MPS | match |",
            "|---|---|---|---|---|---|",
        ]
        all_ok = True
        for N, d, chi in [(4, 2, 3), (6, 2, 4), (2, 2, 5), (5, 3, 2)]:
            psi = qtn.MPS_rand_state(L=N, bond_dim=chi, phys_dim=d)
            actual = sum(t.size for t in psi.tensors)
            got = mps_param_count(N, d, chi)
            ok = got == actual
            all_ok = all_ok and ok
            rows.append(f"| {N} | {d} | {chi} | {got} | {actual} | {'✅' if ok else '❌'} |")
        verdict = "**All checks passed ✅**" if all_ok else "**Something's off ❌**"
        return mo.vstack([mo.md("\n".join(rows)), mo.md(verdict)])

    try:
        _result = _run()
    except NotImplementedError as _e:
        _result = mo.md(f"*Not implemented yet: {_e}*")
    _result
    return


@app.cell(hide_code=True)
def _(mo):
    ex2_solution = mo.ui.run_button(label="💡 Show solution 2")
    ex2_solution
    return (ex2_solution,)


@app.cell(hide_code=True)
def _(ex2_solution, mo):
    mo.stop(not ex2_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def mps_param_count(N, d, chi):
        return 2 * d * chi + (N - 2) * d * chi**2
    ```

    The check compares this against `quimb`'s own
    `qtn.MPS_rand_state(L=N, bond_dim=chi, phys_dim=d)`, summing each
    tensor's `.size` rather than trusting axis order (different `quimb`
    versions lay boundary-tensor axes out differently, `(d, χ)` in some,
    `(χ, d)` in others, the *count* of numbers is what's invariant, and
    the only thing this formula claims).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this buys you

    Fix $\chi$ and grow $N$: the dense statevector needs $d^N$ numbers,
    exponential in $N$. The MPS above needs
    $2d\chi + (N-2)d\chi^2$, **linear** in $N$ for fixed $\chi$. That gap
    is the entire reason tensor networks are useful at all, and it's also
    the catch: a fixed $\chi$ can only represent states whose entanglement
    across every cut of the chain is bounded by $\chi$. Push a state past
    that (e.g. deep, highly-entangling circuits) and no fixed-$\chi$ MPS
    represents it exactly any more, you're back to trading accuracy for
    the polynomial scaling, exactly the "cutoff" and "max_bond_dim"
    knobs the reference notebook's §"From TN to QC" wires up in
    `default.tensor`.
    """)
    return


@app.cell
def _(np, plt):
    _Ns = np.arange(2, 21)
    _d, _chi = 2, 4

    _dense = _d ** _Ns.astype(float)
    _mps = 2 * _d * _chi + np.maximum(_Ns - 2, 0) * _d * _chi**2

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(_Ns, _dense, "o-", color="#c44e52", label=f"dense: $d^N$ ($d={_d}$)")
    ax.plot(_Ns, _mps, "o-", color="#4c72b0", label=f"MPS: params($N,d,\\chi$) ($\\chi={_chi}$)")
    ax.set_yscale("log")
    ax.set_xlabel("sites $N$")
    ax.set_ylabel("numbers stored")
    ax.set_title("Exponential vs. linear: the whole point of a fixed bond dimension")
    ax.legend()
    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: from a TN tensor to a circuit gate

    The reference notebook's §"From TN to QC" replaces each tensor of a
    network with a unitary, then implements that unitary with a
    variational circuit template. The simplest possible instance of that
    step is a single-qubit tensor: a $2\times 2$ unitary $U$, exactly
    what's left at one site of an MPS once every other tensor has been
    contracted away. To actually build a circuit you need to turn $U$
    into gate parameters, and the standard first move is a **Pauli
    decomposition**.

    Every $2\times 2$ complex matrix decomposes uniquely in the Pauli
    basis $\{I, X, Y, Z\}$:

    $$U = c_I I + c_X X + c_Y Y + c_Z Z$$

    because the four Pauli matrices are pairwise orthogonal under the
    Hilbert–Schmidt inner product $\langle A, B\rangle = \mathrm{Tr}(A^\dagger B)$,
    with $\mathrm{Tr}(P^2) = 2$ for each $P$. Multiplying both sides by a
    Pauli matrix $P$ and tracing isolates its own coefficient (using
    $P^\dagger = P$, Pauli matrices are Hermitian):

    $$c_P = \tfrac{1}{2}\,\mathrm{Tr}(P\,U)$$

    Four inner products, no eigendecomposition required. **Implement
    `pauli_decompose(U)`** for a $2\times 2$ numpy array `U`, returning a
    dict `{"I": c_I, "X": c_X, "Y": c_Y, "Z": c_Z}` using that formula,
    against the `PAULI` matrices defined above.
    """)
    return


@app.function
def pauli_decompose(U):
    """Coefficients of U in the {I, X, Y, Z} basis:
    U = c_I*I + c_X*X + c_Y*Y + c_Z*Z.

    TODO: see the exercise text above. c_P = Tr(P @ U) / 2, for each P in
    the PAULI dict.
    """
    raise NotImplementedError("Exercise 3: implement pauli_decompose(U)")


@app.cell(hide_code=True)
def _(mo):
    ex3_check = mo.ui.run_button(label="✅ Check Exercise 3")
    ex3_check
    return (ex3_check,)


@app.cell(hide_code=True)
def _(PAULI, ex3_check, mo, np):
    mo.stop(
        not ex3_check.value,
        mo.md("*Fill in `pauli_decompose` above, then click the button.*"),
    )

    def _random_unitary(rng):
        # Haar-random 2x2 unitary: QR of a complex Ginibre matrix, with
        # the diagonal of R rephased so Q is uniform (not just orthogonal).
        Z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        Q, R = np.linalg.qr(Z)
        phases = np.diagonal(R) / np.abs(np.diagonal(R))
        return Q * phases

    def _run():
        rng = np.random.default_rng(0)
        named = {
            "Hadamard": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
            "S": np.array([[1, 0], [0, 1j]], dtype=complex),
            "T": np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex),
        }
        examples = list(named.items()) + [
            (f"random #{i}", _random_unitary(rng)) for i in range(3)
        ]
        rows = ["| gate | reconstructs U | $\\sum_P |c_P|^2 = 1$ |", "|---|---|---|"]
        all_ok = True
        for name, U in examples:
            coeffs = pauli_decompose(U)
            recon = sum(coeffs[p] * PAULI[p] for p in "IXYZ")
            recon_ok = np.allclose(recon, U)
            norm_ok = np.isclose(sum(abs(coeffs[p]) ** 2 for p in "IXYZ"), 1.0)
            all_ok = all_ok and recon_ok and norm_ok
            rows.append(
                f"| {name} | {'✅' if recon_ok else '❌'} | {'✅' if norm_ok else '❌'} |"
            )
        verdict = (
            "**All checks passed ✅** reconstruction matches, and the "
            "coefficients always land on the unit sphere in $\\mathbb{R}^4$ "
            "(well, $\\mathbb{C}^4$ with a real-norm constraint), exactly "
            "what unitarity of $U$ forces."
            if all_ok
            else "**Something's off ❌** check the rows marked ❌ above."
        )
        return mo.vstack([mo.md("\n".join(rows)), mo.md(verdict)])

    try:
        _result = _run()
    except NotImplementedError as _e:
        _result = mo.md(f"*Not implemented yet: {_e}*")
    _result
    return


@app.cell(hide_code=True)
def _(mo):
    ex3_solution = mo.ui.run_button(label="💡 Show solution 3")
    ex3_solution
    return (ex3_solution,)


@app.cell(hide_code=True)
def _(ex3_solution, mo):
    mo.stop(not ex3_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def pauli_decompose(U):
        return {p: 0.5 * np.trace(mat @ U) for p, mat in PAULI.items()}
    ```

    Nothing fancier than four trace inner products, the Pauli matrices
    being an orthogonal basis under $\mathrm{Tr}(A^\dagger B)$ does all
    the work. The check's second column ($\sum_P |c_P|^2 = 1$) isn't a
    coincidence: $\mathrm{Tr}(U^\dagger U) = 2$ for any unitary, and
    expanding both $U$ and $U^\dagger$ in the same orthogonal basis turns
    that into $2\sum_P |c_P|^2 = 2$. Four complex numbers, one real
    constraint, one more degree of freedom that's pure global phase
    (physically unobservable) exactly the 3 real parameters a
    single-qubit gate is supposed to have.

    The same formula generalizes to $n$-qubit unitaries: replace
    $\{I, X, Y, Z\}$ with all $4^n$ tensor products
    $P_1 \otimes \cdots \otimes P_n$ and divide by $2^n$ instead of $2$,
    exactly what you'd need for a two-site MPS tensor, say. $4^n$ terms
    is exponential, the same story as Exercise 2's $d^N$, which is why
    circuit templates in practice decompose a handful of *local* Pauli
    rotations per gate rather than the whole tensor at once.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From coefficients to a rotation gate

    A Pauli decomposition of $U$ itself gives *complex* coefficients,
    useful for checking unitarity but not yet gate parameters. The
    cleaner route: since $U$ is unitary, it's $U = e^{iH}$ for some
    Hermitian generator $H$ (its matrix logarithm, extracted below via
    eigendecomposition). Pauli-decomposing $H$ *instead* of $U$ gives
    **real** coefficients $h_I, h_X, h_Y, h_Z$ (Pauli matrices are
    Hermitian, so $\mathrm{Tr}(P H)$ is always real for Hermitian $H$),
    and

    $$U = e^{iH} = e^{i h_I}\Big(\cos\theta\, I + i\sin\theta\,(\mathbf{n}\cdot\boldsymbol{\sigma})\Big),
    \qquad \theta = |\mathbf{h}|,\;\; \mathbf{n} = \mathbf{h}/|\mathbf{h}|$$

    with $\mathbf{h} = (h_X, h_Y, h_Z)$. That's a global phase $h_I$
    times a rotation by angle $2\theta$ about axis $\mathbf{n}$, i.e.
    **exactly** the axis and angle you'd hand to a hardware-native
    single-qubit rotation gate (up to the sign convention your circuit
    framework uses for $R_{\mathbf n}(\phi)$). The plot below runs this
    on the Hadamard gate as a concrete, visual example.
    """)
    return


@app.cell
def _(PAULI, np, plt):
    def _decompose(M):
        return {p: 0.5 * np.trace(mat @ M) for p, mat in PAULI.items()}

    _U = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)  # Hadamard
    _coeffs = _decompose(_U)

    _eigvals, _eigvecs = np.linalg.eig(_U)
    _H = _eigvecs @ np.diag(np.angle(_eigvals)) @ np.linalg.inv(_eigvecs)
    _H = (_H + _H.conj().T) / 2  # clean up float noise; H must be Hermitian
    _h = _decompose(_H)
    _axis = np.real([_h["X"], _h["Y"], _h["Z"]])
    _theta = np.linalg.norm(_axis)
    _axis_unit = _axis / _theta

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))

    ax1.bar(list(_coeffs), [abs(v) for v in _coeffs.values()], color="#4c72b0")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("$|c_P|$")
    ax1.set_title("Hadamard: $U = \\sum_P c_P P$")

    ax2.bar(["$n_x$", "$n_y$", "$n_z$"], _axis_unit, color="#55a868")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_title(f"generator axis $\\mathbf{{n}}$ (rotation angle $2\\theta={2*_theta:.3f}$ rad)")

    fig2.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading the right-hand bars: Hadamard's rotation axis is
    $\mathbf{n} \propto -(X+Z)$, and the angle is $2\theta = \pi$, a
    half-turn of the Bloch sphere about the $-(x+z)/\sqrt2$ direction (up
    to the global phase $e^{ih_I}$, which no measurement can see). That
    triple, axis plus angle, is precisely the parameterization
    tensor-network-to-circuit templates (and the PennyLane `MERA`/`MPS`
    templates the reference notebook links in §"From TN to QC") hand off
    to a variational single-qubit gate, one Pauli decomposition per
    tensor in the network.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `quimb` already ships this: `qu.pauli_decomp`

    Exercise 3 built `pauli_decompose` from scratch to see the Hilbert–
    Schmidt formula behind it. `quimb` ships the same computation as
    `qu.pauli_decomp`, decomposing any operator against `{I, X, Y, Z}`
    (or their $n$-qubit tensor products, for operators on more than one
    qubit, exactly the generalization the Exercise 3 solution mentioned).
    Pass `mode="c"` to get the coefficients back as a dict instead of
    just printing them; leave `mode="p"` (the default) and it prints the
    same thing, dominant terms first.
    """)
    return


@app.cell
def _(qu):
    _H_gate = qu.hadamard()
    qu.pauli_decomp(_H_gate, mode="c")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Same `X: 0.707, Z: 0.707` split as the bar chart above (`I` and `Y`
    drop out, both ~0), just without having to write the trace formula
    by hand. Worth knowing it exists, but Exercise 3 was still worth
    doing by hand: `qu.pauli_decomp` hands you the *what*, the loop over
    $\mathrm{Tr}(P U)$ is the *why*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Building H and CNOT from IQM's native gates, with quimb alone

    `Session 2` runs circuits on IQM hardware, whose native gate set is
    just two gates: **PRX**$(\theta, \varphi)$, a rotation by $\theta$
    about the equatorial axis $(\cos\varphi, \sin\varphi, 0)$,

    $$\mathrm{PRX}(\theta,\varphi) = \begin{pmatrix}
    \cos\frac\theta2 & -i e^{-i\varphi}\sin\frac\theta2 \\[2pt]
    -i e^{i\varphi}\sin\frac\theta2 & \cos\frac\theta2
    \end{pmatrix}$$

    and the two-qubit **CZ**. No external transpiler needed to see how
    the `H` and `CNOT` built below (the exact gates the next section
    uses to build `psi0`) reduce to that set, `quimb`'s own
    `qu.pauli_decomp` plus the eigendecomposition trick from Exercise
    3's "From coefficients to a rotation gate" already say enough.
    """)
    return


@app.cell
def _(np):
    def prx(theta, phi):
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array(
            [
                [c, -1j * np.exp(-1j * phi) * s],
                [-1j * np.exp(1j * phi) * s, c],
            ]
        )

    return (prx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### A two-gate recipe for any axis in the $XZ$-plane

    Exercise 3 found Hadamard's own rotation axis, $-(X+Z)/\sqrt2$, has
    a nonzero $Z$-component, so it can't sit on PRX's equator, no single
    PRX reaches it. Two can, though. Direct matrix multiplication of two
    elementary rotations gives, for *any* angle $\alpha$:

    $$R_X(\pi)\,R_Y(2\alpha) \;=\; -i\big(\cos\alpha\,X + \sin\alpha\,Z\big)$$

    the right-hand side is exactly a half-turn (rotation angle $\pi$)
    about the axis $(\cos\alpha, 0, \sin\alpha)$, any axis in the
    $XZ$-plane, reached just by choosing $\alpha$. Since
    $R_Y(2\alpha) = \mathrm{PRX}(2\alpha, \pi/2)$ and
    $R_X(\pi) = \mathrm{PRX}(\pi, 0)$, that's a two-PRX recipe for *any*
    $XZ$-plane half-turn, checked below across a handful of random
    $\alpha$ before trusting it on Hadamard specifically.
    """)
    return


@app.cell
def _(PAULI, np, prx):
    def _check_lemma():
        rng = np.random.default_rng(0)
        ok = True
        for alpha in rng.uniform(0, 2 * np.pi, size=5):
            target = -1j * (np.cos(alpha) * PAULI["X"] + np.sin(alpha) * PAULI["Z"])
            got = prx(np.pi, 0.0) @ prx(2 * alpha, np.pi / 2)
            ok = ok and np.allclose(got, target)
        return ok

    assert _check_lemma()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 4: probability of a measurement outcome

    Exercises 1-3 worked with tensors in isolation, a contraction here,
    a decomposition there. This one puts them to work on an actual
    circuit: build a small entangled state with explicit gates (below),
    then ask `quimb` the one question every tensor in that chain was
    contracted together to answer, how likely is a given measurement
    outcome?
    """)
    return


@app.cell
def _(qtn, qu):
    # some operators to apply
    H = qu.hadamard()
    CNOT = qu.controlled("not")

    # setup an intitial register of qubits
    n = 5
    psi0 = qtn.MPS_computational_state("0" * n, tags="PSI0")

    # apply hadamard to first site
    psi0.gate_(H, 1, tags="H")

    # apply CNOT by qubit pairs
    for i in range(0, n-1, 1):
        psi0.gate_(CNOT, (i, i + 1), tags="CNOT")
    return (psi0,)


@app.cell
def _(psi0):
    psi0.draw(color=["PSI0", "H", "CNOT"], show_tags=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From state to probability

    `psi0` above is an MPS over `n=5` qubits, built by gating a
    computational-basis state exactly like a real circuit would. Reading
    out a measurement probability needs $P(b) = |\langle b | \psi\rangle|^2$
    for a computational-basis bitstring $b$, the same "contract
    everything down to one number" idea Exercise 1 did by hand for two
    tensors, `quimb` just runs it over the whole chain.
    `MatrixProductState.amplitude(b)` returns $\langle b | \psi\rangle$
    given `b` as a sequence of `0`/`1` ints, one per site, in site order.

    **Implement `bitstring_probability(psi, bitstring)`** for an MPS
    `psi` and a `bitstring` given as a string of `'0'`/`'1'` characters
    (e.g. `"01011"`), returning $|\langle b | \psi \rangle|^2$ as a real
    float.
    """)
    return


@app.function
def bitstring_probability(psi, bitstring):
    """|<bitstring|psi>|^2 for an MPS `psi` and a computational-basis
    `bitstring` given as a string of '0'/'1' characters, one per site.

    TODO: see the exercise text above. psi.amplitude(...) wants a
    sequence of ints, not a string of characters.
    """
    raise NotImplementedError(
        "Exercise 4: implement bitstring_probability(psi, bitstring)"
    )


@app.cell(hide_code=True)
def _(mo):
    ex4_check = mo.ui.run_button(label="✅ Check Exercise 4")
    ex4_check
    return (ex4_check,)


@app.cell(hide_code=True)
def _(ex4_check, mo, np, psi0):
    mo.stop(
        not ex4_check.value,
        mo.md("*Fill in `bitstring_probability` above, then click the button.*"),
    )

    def _run():
        dense = psi0.to_dense().reshape(-1)
        bitstrings = ["00000", "01111", "00001", "11111", "10101"]
        rows = [
            "| bitstring | your answer | \\|⟨b\\|ψ⟩\\|² from `to_dense()` | match |",
            "|---|---|---|---|",
        ]
        all_ok = True
        for b in bitstrings:
            want = abs(dense[int(b, 2)]) ** 2
            got = bitstring_probability(psi0, b)
            ok = np.isclose(got, want)
            all_ok = all_ok and ok
            rows.append(f"| {b} | {got:.4f} | {want:.4f} | {'✅' if ok else '❌'} |")
        verdict = "**All checks passed ✅**" if all_ok else "**Something's off ❌**"
        return mo.vstack([mo.md("\n".join(rows)), mo.md(verdict)])

    try:
        _result = _run()
    except NotImplementedError as _e:
        _result = mo.md(f"*Not implemented yet: {_e}*")
    _result
    return


@app.cell(hide_code=True)
def _(mo):
    ex4_solution = mo.ui.run_button(label="💡 Show solution 4")
    ex4_solution
    return (ex4_solution,)


@app.cell(hide_code=True)
def _(ex4_solution, mo):
    mo.stop(not ex4_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def bitstring_probability(psi, bitstring):
        b = [int(c) for c in bitstring]
        return abs(psi.amplitude(b)) ** 2
    ```

    `amplitude` does exactly what §"Contraction" of the reference
    notebook and Exercise 1 did explicitly: it contracts the whole MPS
    chain against a product state $|b\rangle$ (a rank-1 tensor at every
    site) down to a single scalar, $\langle b | \psi \rangle$. Squaring
    its magnitude gives the Born-rule probability.

    Only two bitstrings actually carry weight, `"00000"` and `"01111"`,
    each at `0.5`. Qubit 0 never gets entangled: the first CNOT uses
    qubit 0 as its *control*, and since that qubit is still $|0\rangle$
    the gate is a no-op, it stays $|0\rangle$ throughout. The Hadamard on
    qubit 1, followed by the remaining CNOT chain (1→2, 2→3, 3→4),
    entangles qubits 1-4 into $(|0000\rangle + |1111\rangle)/\sqrt2$, so
    the full 5-qubit state is
    $(|00000\rangle + |01111\rangle)/\sqrt2$, a GHZ state hiding inside a
    spectator qubit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scaling up: sampling instead of enumerating

    `psi0` lives on 5 qubits, small enough that `to_dense()` (32
    numbers) could check your answer directly. That stops being an
    option fast, 80 qubits is $2^{80}$ amplitudes, no dense array will
    ever hold that. What *does* scale is what a real circuit does,
    sample it. `qtn.Circuit` builds the same kind of chain, and
    `.sample(k)` draws `k` measurement outcomes without ever forming the
    full statevector, the linear-in-$N$ MPS storage from Exercise 2 at
    work. The circuit below reuses the same recipe (one Hadamard, then a
    CNOT chain) to build an 80-qubit GHZ state, on a randomly shuffled
    qubit ordering to show the wiring doesn't care about qubit labels,
    then counts 100 samples.
    """)
    return


@app.cell
def _(qtn):
    import random

    N = 80
    circ = qtn.Circuit(N)

    # randomly permute the order of qubits
    regs = list(range(N))
    random.shuffle(regs)

    # hamadard on one of the qubits
    circ.apply_gate("H", regs[0])

    # chain of cnots to generate GHZ-state
    for i_r in range(N - 1):
        circ.apply_gate("CNOT", regs[i_r], regs[i_r + 1])
    return (circ,)


@app.cell
def _(circ):
    from collections import Counter

    # sample it 100 times, count results:
    Counter(circ.sample(100))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go from here

    `3.1-Tensor networks.py` covers contraction order (§"Contraction
    order?") next, the same total computation, done in a different
    sequence, can leave wildly different-sized intermediate tensors on
    the table; `3.2` and `3.3` then build on both ideas, using bond
    dimension as an explicit accuracy/efficiency knob for learning tasks.

    We highly recommend checking out the module [quimb.calc](https://quimb.readthedocs.io/en/latest/autoapi/quimb/calc/index.html) for some useful methods you can directly use that do implement some of the techniques seen in these exercises.

    Exercise 3's Pauli decomposition is deliberately the smallest possible
    version of §"From TN to QC": one $2\times2$ unitary, one gate. Real
    tensor-network circuits chain many such tensors (`qtn.MERA`,
    `qtn.MPS`), each one independently decomposed the same way, and let a
    classical optimizer tune the resulting rotation angles, exactly the
    `default.tensor` example the reference notebook builds next.

    Exercise 4's `bitstring_probability` is the same Born-rule
    amplitude-then-square that `Circuit.sample()` runs internally on
    every draw, just called out explicitly instead of hidden inside a
    sampler, and the same MPS machinery `default.tensor` uses under the
    hood in PennyLane.
    """)
    return


if __name__ == "__main__":
    app.run()
