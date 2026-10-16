# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
#     "qiskit==2.5.2",
#     "qiskit-aer==0.17.2",
# ]
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
    # Exercise: why T-count, and not gate count, is what blows up

    This notebook shows *that* classical simulation cost tracks the T-count $t$
    rather than the qubit count $n$ or the total gate count; this exercise
    has you **measure that blowup yourself, on your own machine**, and
    connect the number you see to the reason it happens.

    1. **Measure** doping (`t_count`), the same way the reference notebook
       does.
    2. **Build** random T-doped circuits (`t_doped_clifford`).
    3. **Implement** the Bravyi–Gosset stabilizer-rank bound
       $\chi(t) \approx 2^{0.23t}$ as a formula, not just a citation.
    4. **Run** a doped circuit through Aer's `extended_stabilizer` method,
       the simulator whose cost this bound governs.
    5. **Compare** a small and a larger T-count head to head and watch the
       wall-clock gap between them, the capstone.

    Same shape as before: a stub function to complete, a **✅ Check**
    button, and a **💡 Show solution** button, collapsed by default. Later
    exercises keep internal reference copies of earlier gadgets, so the
    notebook stays runnable end to end no matter where you are.

    > One heads-up before you start: the **very first** `transpile(...)`
    > call anywhere in this notebook (almost certainly Exercise 1's Check
    > button) pays a one-time ~15-20s cost while Qiskit builds its
    > gate-equivalence library in memory. That cost is paid once per
    > notebook session, not once per call, everything after it is
    > millisecond-fast. It is not part of what any exercise here is
    > measuring; don't let it read as "T=0 is already slow."
    """)
    return


@app.cell
def _():
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    from qiskit_aer import AerSimulator

    return QuantumCircuit, Statevector, np, plt, time, transpile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools you get for free

    - `CLIFFORD_T_BASIS`: the gate set (`h, s, sdg, cx, t, tdg`) every
      function below transpiles into before counting or simulating, so
      "T-count" and "Clifford+T basis" mean the same thing throughout.
    """)
    return


@app.cell
def _():
    CLIFFORD_T_BASIS = ["h", "s", "sdg", "cx", "t", "tdg"]
    return (CLIFFORD_T_BASIS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Warm-up: dense simulation doesn't care about $t$ at all

    Before touching T-count, look at the method the reference notebook
    uses for everything up through its §6: plain dense statevector
    simulation. Its cost is $2^n$ regardless of $t$, by construction, a
    dense state vector has one amplitude per basis state no matter how it
    was prepared. The cell below times it for a few values of $n$, with a
    handful of T gates thrown in but held **fixed** as $n$ grows, to make
    that visible: the curve climbs with $n$ and would look identical if
    every `t()` below were deleted, or tripled.

    This is the contrast Exercise 4 and the capstone are set up against:
    a simulator whose cost curve is flat in $t$ and steep in $n$, versus
    one that is roughly the reverse.
    """)
    return


@app.cell
def _(QuantumCircuit, Statevector, time):
    def _some_circuit(n):
        """Any circuit at all: a handful of Clifford gates plus a fixed
        number of T gates. Its exact content doesn't matter here, only
        that it has n qubits and that the T-count (3, always) does not
        grow with n, this cell is about n, not t."""
        qc = QuantumCircuit(n)
        for q in range(n):
            qc.h(q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
        for q in range(min(n, 3)):
            qc.t(q)
        return qc

    warmup_ns = [4, 6, 8, 10, 12]
    warmup_times = []
    for _n in warmup_ns:
        _qc = _some_circuit(_n)
        _start = time.perf_counter()
        Statevector.from_instruction(_qc)
        warmup_times.append(time.perf_counter() - _start)
    return warmup_ns, warmup_times


@app.cell
def _(plt, warmup_ns, warmup_times):
    fig_warmup, _ax = plt.subplots(figsize=(6.5, 3.2))
    _ax.plot(warmup_ns, warmup_times, "o-", color="#4c72b0")
    _ax.set_xlabel("qubits $n$  (T-count fixed at $t=3$)")
    _ax.set_ylabel("wall time (s)")
    _ax.set_yscale("log")
    _ax.set_title("Dense statevector simulation: grows with $n$, not $t$")
    fig_warmup.tight_layout()
    fig_warmup
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: Measure the doping level

    T-count is only meaningful relative to a gate set, so measure it the
    way the reference notebook does: decompose into `CLIFFORD_T_BASIS`
    first, then count. **Implement `t_count(circuit)`** so it transpiles
    `circuit` into that basis at `optimization_level=0` (no cancellation,
    so the number you get is *this circuit's* T-count, not an optimizer's
    best guess) and returns the number of `t` plus `tdg` gates in the
    result.
    """)
    return


@app.function
def t_count(circuit):
    """Number of T / T-dagger gates after decomposition into Clifford+T.

    TODO: transpile `circuit` into CLIFFORD_T_BASIS at optimization_level=0,
    then return ops.get("t", 0) + ops.get("tdg", 0) from its count_ops().
    """
    raise NotImplementedError("Exercise 1: implement t_count(circuit)")


@app.cell(hide_code=True)
def _(mo):
    ex1_check = mo.ui.run_button(label="✅ Check Exercise 1")
    ex1_check
    return (ex1_check,)


@app.cell(hide_code=True)
def _(CLIFFORD_T_BASIS, QuantumCircuit, ex1_check, mo, transpile):
    mo.stop(
        not ex1_check.value,
        mo.md("*Fill in `t_count` above, then click the button.*"),
    )

    def _reference(circuit):
        decomposed = transpile(
            circuit, basis_gates=CLIFFORD_T_BASIS, optimization_level=0
        )
        ops = decomposed.count_ops()
        return ops.get("t", 0) + ops.get("tdg", 0)

    def _build(k):
        # Each T-type gate on its own fresh qubit: two T's back-to-back on
        # the *same* idle qubit would correctly collapse to a single S
        # gate even at optimization_level=0 (T*T=S is just true, not an
        # optional cancellation), which would make "k gates in, k gates
        # out" the wrong expectation. Distinct qubits sidestep that.
        qc = QuantumCircuit(k + 2)
        qc.h(0)
        qc.cx(0, 1)
        for i in range(k):
            (qc.t if i % 2 == 0 else qc.tdg)(2 + i)
        qc.s(1)
        return qc

    def _run():
        rows = ["| built T-count | your t_count | matches |", "|---|---|---|"]
        all_ok = True
        for k in (0, 1, 3, 5):
            qc = _build(k)
            got = t_count(qc)
            want = _reference(qc)
            ok = got == want == k
            all_ok = all_ok and ok
            rows.append(f"| {k} | {got} | {'✅' if ok else '❌'} |")
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
    def t_count(circuit):
        decomposed = transpile(
            circuit, basis_gates=CLIFFORD_T_BASIS, optimization_level=0
        )
        ops = decomposed.count_ops()
        return ops.get("t", 0) + ops.get("tdg", 0)
    ```

    `optimization_level=0` matters: higher levels actively hunt for
    cancellations, commuting a `t` past other gates to pair it up with a
    distant `tdg` it wasn't originally adjacent to, and would report a
    *smaller* count than the circuit you actually built. What
    `optimization_level=0` does **not** do, at any level, is let you avoid
    the honest arithmetic of gates that already share a wire back to back:
    two `t`'s in a row on an idle qubit *are* one `s` ($T^2 = S$), four are
    the identity's Clifford cousin $Z$, that's not an optimizer's opinion,
    it's the gate's definition. `t_doped_clifford` (Exercise 2) sidesteps
    the whole question by wrapping every T gate in a pair of
    `circuit.barrier()`s, a hard boundary this consolidation can never
    cross, rather than trying to reason out which neighbouring gates are
    "safe."
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Build a random T-doped circuit

    You could draws each Clifford block from
    `random_clifford`, the exact Haar-random sampler. This exercise uses a
    **deliberately cruder** stand-in: a short random sequence of `h`, `s`,
    `sdg`, `cx` gates as one "Clifford layer". It is not uniformly random
    over the Clifford group, and that's fine, nothing below needs it to
    be. What it buys back is speed: every gate it produces is already in
    `CLIFFORD_T_BASIS`, so nothing downstream ever needs to *decompose* it
    into that basis, only *count* or *run* it. (You'll see exactly what
    that decomposition step costs once you reach Exercise 4.)

    One trap worth naming before you write this: whatever a layer happens
    to do right before or after a `t()` call, on the *same* qubit, can
    merge with it once Exercise 1's `transpile` consolidates each wire's
    run of single-qubit gates, and not always by shrinking, `s` followed
    by `t` on one qubit is $S \cdot T = T^3$, which needs **three** `t`/`tdg`
    gates to write back out, silently *inflating* your T-count instead.
    Rather than reason case by case about which random combinations are
    safe, wall the T gate off completely: **implement
    `t_doped_clifford(n, t, seed, layer_gates=3)`** using `circuit.barrier()`
    as a hard boundary the transpiler's consolidation pass will never
    cross, the same barriers the reference notebook's §1 uses, just load-
    bearing here instead of cosmetic:

    - seed a generator with `np.random.default_rng(seed)`;
    - write a helper that appends one layer: `layer_gates` times, pick a
      gate name uniformly from `["h", "s", "sdg", "cx"]` via `rng.choice`;
      for `"cx"` pick two distinct random qubits with
      `rng.choice(n, size=2, replace=False)` (skip it, or fall back to a
      single-qubit gate, when `n == 1`), otherwise pick one random qubit
      with `rng.integers(n)` and apply that gate to it;
    - loop `t` times: append one layer, then `circuit.barrier()`, then one
      `t()` on a random qubit, then `circuit.barrier()` again, sandwiching
      every T gate between two barriers regardless of what the random
      layer around it did;
    - append one final layer after the loop.
    """)
    return


@app.function
def t_doped_clifford(n, t, seed=None, layer_gates=3):
    """C_t · T · C_{t-1} · ... · T · C_0 on n qubits, with exactly t T gates,
    where each "C_i" is a short random layer of h/s/sdg/cx gates.

    TODO: see the exercise text above for the exact recipe.
    """
    raise NotImplementedError("Exercise 2: implement t_doped_clifford(n, t, seed)")


@app.cell(hide_code=True)
def _(mo):
    ex2_check = mo.ui.run_button(label="✅ Check Exercise 2")
    ex2_check
    return (ex2_check,)


@app.cell(hide_code=True)
def _(CLIFFORD_T_BASIS, ex2_check, mo, transpile):
    mo.stop(
        not ex2_check.value,
        mo.md("*Fill in `t_doped_clifford` above, then click the button.*"),
    )

    def _count_t(circuit):
        decomposed = transpile(
            circuit, basis_gates=CLIFFORD_T_BASIS, optimization_level=0
        )
        ops = decomposed.count_ops()
        return ops.get("t", 0) + ops.get("tdg", 0)

    def _run():
        rows = [
            "| n | t | seed | T-count correct | reproducible | verdict |",
            "|---|---|---|---|---|---|",
        ]
        all_ok = True
        for n, t, seed in [(3, 2, 1), (4, 5, 7), (2, 0, 3)]:
            qc_a = t_doped_clifford(n, t, seed=seed)
            qc_b = t_doped_clifford(n, t, seed=seed)
            count_ok = _count_t(qc_a) == t
            repro_ok = qc_a == qc_b
            ok = count_ok and repro_ok
            all_ok = all_ok and ok
            rows.append(
                f"| {n} | {t} | {seed} | {'✅' if count_ok else '❌'} | "
                f"{'✅' if repro_ok else '❌'} | {'✅' if ok else '❌'} |"
            )
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
    def t_doped_clifford(n, t, seed=None, layer_gates=3):
        rng = np.random.default_rng(seed)
        circuit = QuantumCircuit(n, name=f"t_doped_n{n}_t{t}")

        def layer():
            for _ in range(layer_gates):
                gate = rng.choice(["h", "s", "sdg", "cx"])
                if gate == "cx" and n > 1:
                    a, b = (int(q) for q in rng.choice(n, size=2, replace=False))
                    circuit.cx(a, b)
                else:
                    getattr(circuit, "h" if gate == "cx" else gate)(int(rng.integers(n)))

        for _ in range(t):
            layer()
            circuit.barrier()
            circuit.t(int(rng.integers(n)))
            circuit.barrier()
        layer()
        return circuit
    ```

    Every circuit built this way contains **exactly** $t$ T gates and
    nothing else non-Clifford, so `t_count` on its output is not an
    estimate, it is a ground truth you set yourself. That matters for
    everything from here on: the capstone attributes a timing change to
    "T-count went up" only because this function guarantees it, and
    nothing else about the circuit, changed. The two `barrier()` calls
    are structural, not decorative here: without them, whatever the
    random layer happened to leave on a T gate's own qubit would merge
    with it during Exercise 1's `transpile`, sometimes shrinking the
    count, sometimes (as with $S \cdot T = T^3$) inflating it. It also
    matters that every gate here is already Clifford+T, unlike
    `random_clifford().to_circuit()`, which is exactly why this version
    doesn't pay the decomposition cost Exercise 4's solution talks about.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: The stabilizer-rank bound, as a formula

    A stabilizer state is a state a poly-size tableau can represent
    exactly, and by the Gottesman–Knill theorem, propagating that tableau
    through Clifford gates costs $\mathrm{poly}(n)$ *no matter how many*
    Clifford gates there are. A single T gate breaks that: it turns a
    stabilizer state into a **magic state** no single tableau can
    represent, so Aer's `extended_stabilizer` method instead writes the
    state as a weighted sum of $\chi$ stabilizer states (a "stabilizer
    decomposition") and pays a cost that scales with $\chi$, not with $n$.

    Bravyi and Gosset's bound on how large that sum needs to be, after
    their optimisations, is

    $$\chi(t) \;\approx\; 2^{0.23\,t}.$$

    **Implement `stabilizer_rank_bound(t)`** as exactly that formula. It
    is one line, the point of the exercise is reading it carefully enough
    to notice what it does *not* depend on: $n$, circuit depth, or the
    number of Clifford gates. All the blowup lives in $t$ alone.
    """)
    return


@app.function
def stabilizer_rank_bound(t):
    """TODO: return the Bravyi-Gosset bound 2 ** (0.23 * t)."""
    raise NotImplementedError("Exercise 3: implement stabilizer_rank_bound(t)")


@app.cell(hide_code=True)
def _(mo):
    ex3_check = mo.ui.run_button(label="✅ Check Exercise 3")
    ex3_check
    return (ex3_check,)


@app.cell(hide_code=True)
def _(ex3_check, mo):
    mo.stop(
        not ex3_check.value,
        mo.md("*Fill in `stabilizer_rank_bound` above, then click the button.*"),
    )

    def _run():
        rows = ["| t | your χ(t) | reference | match |", "|---|---|---|---|"]
        all_ok = True
        for t in (0, 1, 5, 10, 20):
            want = 2 ** (0.23 * t)
            got = stabilizer_rank_bound(t)
            ok = abs(got - want) < 1e-9
            all_ok = all_ok and ok
            rows.append(f"| {t} | {got:.4f} | {want:.4f} | {'✅' if ok else '❌'} |")
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
    ex3_solution = mo.ui.run_button(label="💡 Show solution 3")
    ex3_solution
    return (ex3_solution,)


@app.cell(hide_code=True)
def _(ex3_solution, mo):
    mo.stop(not ex3_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def stabilizer_rank_bound(t):
        return 2 ** (0.23 * t)
    ```

    Small as it looks, this is the entire theoretical claim: $\chi(t)$ is
    **exponential in $t$**, just with a much smaller base ($2^{0.23}
    \approx 1.17$) than the naive $2^t$ you'd get from expanding every T
    gate's magic state in the computational basis by brute force. Compare
    that to $t = O(\log n) \Rightarrow \chi = \mathrm{poly}(n)$, the
    provably-easy regime from the reference notebook's introduction table,
    a one-line consequence of exactly this formula.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4: Run a doped circuit through the stabilizer-rank simulator

    Time to use the simulator whose cost Exercise 3's bound describes.
    **Implement `run_extended_stabilizer(circuit, shots, seed)`**:

    - copy `circuit` and append `measure_all()` to the copy (don't mutate
      the caller's circuit);
    - transpile that copy into `CLIFFORD_T_BASIS` at `optimization_level=0`
      (Aer's extended-stabilizer backend only recognises Clifford+T gates,
      a generic basis like `rz`/`sx` gets rejected or silently
      approximated);
    - build `AerSimulator(method="extended_stabilizer")` and run it for
      `shots` shots with `seed_simulator=seed`;
    - return the resulting counts dict.
    """)
    return


@app.function
def run_extended_stabilizer(circuit, shots, seed=0):
    """TODO: measure `circuit` `shots` times via Aer's extended_stabilizer
    method and return the counts dict. See the exercise text above."""
    raise NotImplementedError(
        "Exercise 4: implement run_extended_stabilizer(circuit, shots, seed)"
    )


@app.cell(hide_code=True)
def _(mo):
    ex4_check = mo.ui.run_button(label="✅ Check Exercise 4")
    ex4_check
    return (ex4_check,)


@app.cell(hide_code=True)
def _(QuantumCircuit, ex4_check, mo):
    mo.stop(
        not ex4_check.value,
        mo.md("*Fill in `run_extended_stabilizer` above, then click the button.*"),
    )

    def _run():
        # A deterministic stabilizer circuit: |0 1 0>, every shot must agree.
        qc = QuantumCircuit(3)
        qc.x(1)
        shots = 200
        counts = run_extended_stabilizer(qc, shots, seed=0)

        shots_ok = sum(counts.values()) == shots
        keys_ok = all(len(k) == 3 and set(k) <= {"0", "1"} for k in counts)
        deterministic_ok = counts == {"010": shots}

        all_ok = shots_ok and keys_ok and deterministic_ok
        rows = [
            "| check | result |",
            "|---|---|",
            f"| shots sum to {shots} | {'✅' if shots_ok else '❌'} |",
            f"| keys are valid 3-bit strings | {'✅' if keys_ok else '❌'} |",
            f"| deterministic outcome = `010` on every shot | "
            f"{'✅' if deterministic_ok else '❌'} |",
        ]
        verdict = (
            "**All checks passed ✅**"
            if all_ok
            else f"**Something's off ❌** — got `{counts}`"
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
    ex4_solution = mo.ui.run_button(label="💡 Show solution 4")
    ex4_solution
    return (ex4_solution,)


@app.cell(hide_code=True)
def _(ex4_solution, mo):
    mo.stop(not ex4_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def run_extended_stabilizer(circuit, shots, seed=0):
        measured = circuit.copy()
        measured.measure_all()
        tqc = transpile(
            measured, basis_gates=CLIFFORD_T_BASIS, optimization_level=0
        )
        sim = AerSimulator(method="extended_stabilizer")
        result = sim.run(tqc, shots=shots, seed_simulator=seed).result()
        return result.get_counts()
    ```

    Nothing about the *result* here is new, at $t=0$ this backend has to
    agree with the dense statevector method from the warm-up, since both
    are simulating the same circuit exactly. What's different is *how*
    it gets there, and that difference is what the capstone times.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capstone: feel the blowup yourself

    Keep it deliberately small: **fix** $n=3$ and compare just three
    T-counts, $t \in \{0, 3, 6\}$, using your `t_doped_clifford` and your
    `run_extended_stabilizer`, timing each with a few repeats (to average
    out simulator noise, this Monte-Carlo method's runtime isn't perfectly
    smooth run to run) at a handful of shots apiece. Even this small a
    sweep, on ordinary laptop hardware, is enough: $t=0$ finishes near
    instantly, and $t=6$ is already clearly, sometimes dramatically,
    slower, while $n$ and the shot count never move.

    This button should finish in well under a minute (the one-time
    transpile warm-up mentioned at the top aside). If it's still dragging,
    shrink `CAPSTONE_SHOTS` or `CAPSTONE_REPS` below, the qualitative
    point survives at even smaller values.
    """)
    return


@app.cell
def _():
    CAPSTONE_N = 3
    CAPSTONE_TS = [0, 3, 6]
    CAPSTONE_SHOTS = 5
    CAPSTONE_REPS = 3
    return CAPSTONE_N, CAPSTONE_REPS, CAPSTONE_SHOTS, CAPSTONE_TS


@app.cell(hide_code=True)
def _(mo):
    run_capstone = mo.ui.run_button(label="▶ Compare t = 0, 3, 6")
    run_capstone
    return (run_capstone,)


@app.cell(hide_code=True)
def _(
    CAPSTONE_N,
    CAPSTONE_REPS,
    CAPSTONE_SHOTS,
    CAPSTONE_TS,
    mo,
    np,
    plt,
    run_capstone,
    time,
):
    mo.stop(not run_capstone.value, mo.md("*Press the button to run.*"))

    def _run():
        times = []
        for t in CAPSTONE_TS:
            reps = []
            for rep in range(CAPSTONE_REPS):
                qc = t_doped_clifford(CAPSTONE_N, t, seed=1000 * t + rep)
                start = time.perf_counter()
                run_extended_stabilizer(qc, CAPSTONE_SHOTS, seed=rep)
                reps.append(time.perf_counter() - start)
            times.append(float(np.mean(reps)))

        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.bar([str(t) for t in CAPSTONE_TS], times, color="#4c72b0")
        ax.set_yscale("log")
        ax.set_xlabel(f"T-count $t$  (qubits fixed at n = {CAPSTONE_N})")
        ax.set_ylabel(f"mean wall time over {CAPSTONE_REPS} reps (s)")
        ax.set_title("extended_stabilizer runtime vs. T-count, n and shots fixed")
        fig.tight_layout()

        ratio = times[-1] / max(times[0], 1e-9)
        verdict = mo.md(
            f"At $t={CAPSTONE_TS[0]}$: **{times[0]:.3f}s**. At "
            f"$t={CAPSTONE_TS[-1]}$: **{times[-1]:.3f}s**, roughly "
            f"**{ratio:,.0f}×** slower, while $n={CAPSTONE_N}$ and the "
            f"shot count ({CAPSTONE_SHOTS}) never changed. Don't expect "
            "that ratio to land exactly on $2^{0.23 \\times 6} \\approx "
            f"{2**(0.23*6):.1f}$: the asymptotic bound from Exercise 3 is "
            "a worst-case rate for an *optimal* decomposition, this "
            "Monte-Carlo sampler and this particular random circuit can, "
            "and often does, do worse in practice, especially at these "
            "small, noisy sample sizes. The qualitative claim is the one "
            "that matters and the one this cell actually tests: the "
            "slowdown tracks $t$ alone, with $n$ and gate count held "
            "fixed the whole time."
        )
        return mo.vstack([fig, verdict])

    try:
        _result = _run()
    except NotImplementedError as _e:
        _result = mo.md(f"*Finish Exercises 2 and 4 above first: {_e}*")
    _result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why it blows up, in one paragraph

    Clifford gates are free in a very specific sense: the Gottesman–Knill
    theorem lets a classical computer track *any* stabilizer circuit by
    propagating a size-$O(n^2)$ tableau through each gate, at a cost per
    gate that never depends on how many gates came before it. Depth,
    qubit count, gate count, none of that is what this exercise's capstone worry about, and the warm-up above
    showed dense simulation cost tracking $n$ for exactly this reason,
    it never even tries to exploit Clifford structure. A T gate is the one
    operation the tableau *can't* absorb: it turns a stabilizer state into
    a magic state, forcing the simulator to fall back to representing the
    state as a weighted sum of stabilizer terms. Every extra T gate can
    make that sum harder to compress, and Bravyi–Gosset's result is that,
    after their optimisations, the sum's size still has to grow, just
    slowly, like $2^{0.23t}$ rather than the naive $2^t$. That is the
    entire reason fault-tolerant budgets are quoted in **T-count**, not
    gate count: it is the one resource this asymptotic actually depends on.

    For the reasoning behind $\chi(t) \approx 2^{0.23t}$ itself, and for
    the anti-concentration / Porter–Thomas story that sits on top of this
    (a separate question from *cost*, namely what the output *distribution*
    looks like as $t$ grows).
    """)
    return


if __name__ == "__main__":
    app.run()
