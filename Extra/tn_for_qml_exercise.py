# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "pennylane==0.45.0",
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
    # Exercise: TNs for AI and for quantum, side by side

    A hands-on companion to `3.2-Neural Networks as TNs.py` and
    `3.3-QML on TNs.py`. Those notebooks train real classifiers on MNIST
    and digits data, real training loops take real minutes, so this
    exercise skips training entirely and isolates the one mechanic each
    notebook actually depends on:

    1. **AI side (3.2):** the single contraction step an MPS classifier
       repeats at every site, the thing that *is* its forward pass.
    2. **Quantum side (3.3):** that a tensor-network device is a drop-in
       replacement for a statevector device, same circuit code, same
       physics, different engine underneath.

    Same shape as the other `Extra/` exercises: a stub function to
    complete, a **✅ Check** button, and a **💡 Show solution** button,
    collapsed by default.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pennylane as qml

    return (np,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: One step of an MPS classifier's forward pass

    `3.2` encodes each classical feature $x_i \in [0, 1]$ as a unit
    2-vector, $\varphi(x_i) = [\cos(\pi x_i / 2), \sin(\pi x_i / 2)]$,
    then feeds that chain of vectors into a **weight MPS** one site at a
    time. At a bulk site, the running "bond vector" $v$ (one per batch
    element) is updated by contracting it against that site's weight
    tensor $A$ and the local feature $\varphi_i$:

    $$v'_{b,k} = \sum_{j,p} v_{b,j} \, A_{j,p,k} \, \varphi_{b,p}$$

    with $v$ shaped `(B, χ)`, $A$ shaped `(χ, d, χ)`, $\varphi_i$ shaped
    `(B, d)`, and the result $v'$ shaped `(B, χ)`. Run this once per site
    down the whole chain and you have `3.2`'s entire classifier, an MPS
    *is* the model, this contraction *is* the forward pass, there's no
    separate "matrix multiply layer" hiding underneath.

    **Implement `mps_forward_step(v, A, phi)`** as exactly that sum.
    `np.einsum("bj,jpk,bp->bk", v, A, phi)` is the one-line way; explicit
    loops over `b, k, j, p` work too and compute the same thing.
    """)
    return


@app.function
def mps_forward_step(v, A, phi):
    """v: (B, chi), A: (chi, d, chi), phi: (B, d) -> v': (B, chi).

    TODO: see the exercise text above.
    """
    raise NotImplementedError("Exercise 1: implement mps_forward_step(v, A, phi)")


@app.cell(hide_code=True)
def _(mo):
    ex1_check = mo.ui.run_button(label="✅ Check Exercise 1")
    ex1_check
    return (ex1_check,)


@app.cell(hide_code=True)
def _(ex1_check, mo, np):
    mo.stop(
        not ex1_check.value,
        mo.md("*Fill in `mps_forward_step` above, then click the button.*"),
    )

    def _reference(v, A, phi):
        # Deliberately the slow, obviously-correct version: four nested
        # sums, spelling out exactly what the formula says.
        B, chi = v.shape
        _, d, chi2 = A.shape
        out = np.zeros((B, chi2))
        for b in range(B):
            for k in range(chi2):
                s = 0.0
                for j in range(chi):
                    for p in range(d):
                        s += v[b, j] * A[j, p, k] * phi[b, p]
                out[b, k] = s
        return out

    def _run():
        rng = np.random.default_rng(1)
        rows = ["| B | χ | d | matches slow reference |", "|---|---|---|---|"]
        all_ok = True
        for B, chi, d in [(3, 2, 2), (5, 4, 2), (2, 3, 3)]:
            v = rng.normal(size=(B, chi))
            A = rng.normal(size=(chi, d, chi))
            phi = rng.normal(size=(B, d))
            got = mps_forward_step(v, A, phi)
            want = _reference(v, A, phi)
            ok = got.shape == want.shape and np.allclose(got, want)
            all_ok = all_ok and ok
            rows.append(f"| {B} | {chi} | {d} | {'✅' if ok else '❌'} |")
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
    def mps_forward_step(v, A, phi):
        return np.einsum("bj,jpk,bp->bk", v, A, phi)
    ```

    `3.2` renormalises $v$ after every step
    (`v = v / (v.norm(dim=1, keepdim=True) + 1e-8)`) and picks a feature
    map with $\lVert\varphi(x)\rVert = 1$ specifically so this repeated
    contraction doesn't decay or blow up over a long chain, the same
    "many small factors compound" concern from the T-count exercise, just
    guarded against here instead of measured. The very last site swaps
    its second bond for a `n_class`-sized output index, so the same
    contraction, run one more time, produces logits instead of another
    bond vector, that's the whole classifier: no separate "readout
    layer," just one more site of the same chain.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: A tensor-network device is a drop-in replacement

    `3.3`'s punchline is one line: point `qml.device(...)` at
    `"default.tensor"` instead of `"default.qubit"` and the *exact same*
    circuit code runs on a tensor-network simulator underneath, same
    inputs, same outputs, no change to the circuit itself. The circuit
    doesn't know or care which engine is contracting it.

    **Implement `ghz_correlation(n_wires, device_name)`**:

    - build `dev = qml.device(device_name, wires=n_wires)`;
    - define and run a `@qml.qnode(dev)`-decorated circuit that prepares
      an $n$-qubit GHZ state, `Hadamard` on wire `0`, then a `CNOT` chain
      `(0,1), (1,2), ..., (n_wires-2, n_wires-1)`, and returns
      `qml.expval(qml.PauliZ(0) @ qml.PauliZ(n_wires - 1))`;
    - return that expectation value.

    For a GHZ state
    $\frac{1}{\sqrt2}(|0\cdots0\rangle + |1\cdots1\rangle)$, both terms
    have matching parity on every qubit, so $Z_0 Z_{n-1}$ should land
    on exactly $1$, on *either* device, that shared answer is what the
    check compares.
    """)
    return


@app.function
def ghz_correlation(n_wires, device_name):
    """TODO: build an n_wires-qubit GHZ circuit on the named device and
    return <Z_0 Z_{n_wires-1}>. See the exercise text above."""
    raise NotImplementedError(
        "Exercise 2: implement ghz_correlation(n_wires, device_name)"
    )


@app.cell(hide_code=True)
def _(mo):
    ex2_check = mo.ui.run_button(label="✅ Check Exercise 2")
    ex2_check
    return (ex2_check,)


@app.cell(hide_code=True)
def _(ex2_check, mo, np):
    mo.stop(
        not ex2_check.value,
        mo.md("*Fill in `ghz_correlation` above, then click the button.*"),
    )

    def _run():
        rows = [
            "| n_wires | default.qubit | default.tensor | both ≈ 1 and match |",
            "|---|---|---|---|",
        ]
        all_ok = True
        for n in (2, 3, 5):
            a = ghz_correlation(n, "default.qubit")
            b = ghz_correlation(n, "default.tensor")
            ok = np.isclose(a, 1.0, atol=1e-6) and np.isclose(a, b, atol=1e-6)
            all_ok = all_ok and ok
            rows.append(f"| {n} | {a:.6f} | {b:.6f} | {'✅' if ok else '❌'} |")
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
    def ghz_correlation(n_wires, device_name):
        dev = qml.device(device_name, wires=n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(wires=0)
            for w in range(n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(n_wires - 1))

        return circuit()
    ```

    Nothing about `circuit` mentions tensors, bond dimensions, or `quimb`
    anywhere, `default.tensor` reads the *exact same* gate sequence
    `default.qubit` does and, under the hood, builds a tensor network out
    of it (`3.1`'s "From TN to QC" section) instead of a $2^n$-entry
    statevector. That's the entire practical upshot for `3.3`: the same
    circuit swapped onto a bigger device (more qubits, deeper ansatz)
    keeps working, `default.qubit` just gets impractically slow first.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: what a fixed bond dimension can (and can't) hold

    Exercise 2 ran the *same* GHZ circuit on both devices, but only
    `default.tensor` stores the result as an MPS, a chain of tensors
    joined by a **bond dimension** $\chi$. `tensor_networks_exercise.py`
    claims a fixed $\chi$ can only represent states whose entanglement
    across every cut is bounded by $\chi$, worth seeing directly rather
    than taking on faith.

    Slice an $n$-qubit statevector in half, qubits $0..n/2-1$ against
    $n/2..n-1$, and reshape it into a $2^{n/2}\times2^{n/2}$ matrix. Its
    singular values *are* the Schmidt coefficients of that cut, and the
    count of nonzero ones is exactly the bond dimension an MPS needs
    there to be exact, no more, no less. Two $n=12$ states, same cut:

    - **GHZ**, $\frac{1}{\sqrt2}(|0\cdots0\rangle+|1\cdots1\rangle)$,
      the least-entangled state that still isn't a product state.
    - a **Haar-random statevector** (a normalized random complex
      Gaussian vector), about as entangled as any $n$-qubit state can
      be, nothing to exploit anywhere.

    All numpy, one small SVD, nothing trained or sampled.
    """)
    return


@app.cell(hide_code=True)
def _(np):
    import matplotlib.pyplot as plt

    _n = 12  # 2**12 = 4096-entry statevector -> reshapes to a 64x64 matrix: instant
    _half = _n // 2  # cut straight down the middle of the chain

    def _schmidt_probs(state):
        mat = state.reshape(2**_half, 2 ** (_n - _half))
        s = np.linalg.svd(mat, compute_uv=False)
        p = s**2
        return np.sort(p / p.sum())[::-1]

    def _entropy(p):
        p = p[p > 1e-14]
        return -np.sum(p * np.log2(p))

    _rng = np.random.default_rng(0)

    _ghz = np.zeros(2**_n)
    _ghz[0] = _ghz[-1] = 1 / np.sqrt(2)

    _rand = _rng.normal(size=2**_n) + 1j * _rng.normal(size=2**_n)
    _rand /= np.linalg.norm(_rand)

    _p_ghz = _schmidt_probs(_ghz)
    _p_rand = _schmidt_probs(_rand)

    fig3, axes3 = plt.subplots(1, 3, figsize=(13, 3.4))

    axes3[0].bar(range(10), _p_ghz[:10], color="#4c72b0")
    axes3[0].set_yscale("log")
    axes3[0].set_ylim(1e-16, 2)
    axes3[0].set_xlabel("Schmidt index")
    axes3[0].set_ylabel("$p_i$ (normalized, sorted)")
    axes3[0].set_title(f"GHZ, cut at qubit {_half}\nentropy = {_entropy(_p_ghz):.3f} bits")

    axes3[1].bar(range(2**_half), _p_rand, color="#c44e52", width=1.0)
    axes3[1].set_xlabel("Schmidt index")
    axes3[1].set_title(
        f"Haar-random, cut at qubit {_half}\nentropy = {_entropy(_p_rand):.3f} bits (max {_half})"
    )

    _chis = np.arange(1, 2**_half + 1)
    _err_ghz = [1.0 - _p_ghz[:c].sum() for c in _chis]
    _err_rand = [1.0 - _p_rand[:c].sum() for c in _chis]
    axes3[2].plot(_chis, _err_ghz, "o-", color="#4c72b0", label="GHZ")
    axes3[2].plot(_chis, _err_rand, "o-", color="#c44e52", label="Haar-random")
    axes3[2].set_yscale("log")
    axes3[2].set_ylim(1e-16, 2)
    axes3[2].set_xlabel(r"bond dimension $\chi$ kept")
    axes3[2].set_ylabel(r"truncation error $1-\sum_{i<\chi}p_i$")
    axes3[2].set_title("cost of under-sizing $\\chi$")
    axes3[2].legend()

    fig3.tight_layout()
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reading left to right:

    - **GHZ's spectrum** (left) has exactly two nonzero entries, each
      $0.5$ after normalizing: $S=1$ bit of entanglement across *any*
      cut, the minimum a non-product state can have. A $\chi=2$ MPS
      stores this exactly, at every bond, for any $n$, which is why
      Exercise 2's `default.tensor` run never needed $\chi$ to grow with
      $n$.
    - **The Haar-random spectrum** (middle) is close to flat across all
      $2^{n/2}$ Schmidt values, entropy near its theoretical maximum of
      $n/2$ bits, "maximally entangled" in the literal sense: no cut is
      any easier to compress than another, nothing here favors a small
      $\chi$.
    - **The truncation-error curve** (right) makes the contrast
      concrete: GHZ's error hits *exactly* zero the moment $\chi$
      reaches 2 and stays there; the random state's error only reaches
      zero once $\chi$ reaches the full $2^{n/2}$, decaying smoothly the
      whole way, no shortcut $\chi$ to aim for.

    Real circuits live somewhere between these two curves. Shallow or
    structured states (GHZ, most variational ansätze early in training)
    sit close to the GHZ curve, a modest $\chi$ captures them almost
    exactly. Deep, highly-scrambling circuits drift toward the
    random-state curve, and that's exactly where `default.tensor`'s
    `max_bond_dim`/`cutoff` knobs (`3.1`'s §"From TN to QC") start
    trading accuracy for tractability, right where this curve stops
    looking flat.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go from here

    `3.2` scales Exercise 1 up to a real MNIST classifier and sweeps
    $\chi$ against test accuracy, watch how few parameters (§"How many
    numbers does an MPS actually store?" in `tensor_networks_exercise.py`)
    it takes to stay competitive. `3.3` scales Exercise 2 up to a trained
    QNN and a real encoding circuit, and shows the same device swap
    applied mid-training, along with the honest caveat: bigger circuits
    on a TN device don't sidestep barren plateaus or overfitting, they
    only make each individual run cheaper to attempt.
    """)
    return


if __name__ == "__main__":
    app.run()
