# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
#     "qiskit==2.5.2",
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
    # Exercise: building Boolean penalty Hamiltonians, step by step

    This is a hands on exercise on how to:

    1. **Implement** a gadget from its formula (AND).
    2. **Reuse** it to build a new one (OR, via De Morgan) instead of
       inventing one from scratch.
    3. **Substitute** Boolean variables for Pauli operators (QUBO → Ising).
    4. **Tune** a penalty weight against a competing objective.
    5. **Simulate**, with Qiskit circuits, which bitstring actually wins.

    Every exercise follows the same shape: a stub function to complete, a
    **✅ Check** button that runs the same tests as the reference solution,
    and a **💡 Show solution** button that reveals a worked answer, collapsed
    by default so it doesn't spoil the attempt. Later exercises never depend
    on your unfinished code: internal reference gadgets keep the notebook
    runnable end to end no matter where you are, so feel free to skip around.
    """)
    return


@app.cell
def _():
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    return SparsePauliOp, itertools, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tools you get for free

    Four small utilities, all reused throughout. None of them are the
    exercise, they're the workbench:

    - `evaluate(polynomial, assignment)`: evaluate a `{frozenset: coeff}`
      multilinear polynomial at a 0/1 assignment.
    - `mobius_coefficients(f, n)`: recover a polynomial's coefficients from
      its truth table by inclusion-exclusion (see the reference notebook's
      §1 for why this is *the* unique square-free representation).
    - `spectrum(polynomial, n)` / `bits_of(index, n)`: evaluate a polynomial
      on all $2^n$ assignments at once, for plotting and brute-force checks.
    - `poly_to_str(polynomial, names)`: pretty-print a coefficient dict for
      the markdown cells below.
    """)
    return


@app.cell
def _(itertools, np):
    def evaluate(polynomial, assignment):
        """Evaluate a multilinear polynomial at a dict/list of 0-1 values."""
        total = 0.0
        for term, coeff in polynomial.items():
            value = coeff
            for var in term:
                value *= assignment[var]
            total += value
        return total

    def mobius_coefficients(f, n):
        """Multilinear coefficients of f: {0,1}^n -> R via inclusion-exclusion."""
        coeffs = {}
        for size in range(n + 1):
            for combo in itertools.combinations(range(n), size):
                S = frozenset(combo)
                total = 0.0
                for t_size in range(len(S) + 1):
                    for T in itertools.combinations(sorted(S), t_size):
                        assignment = [1 if i in T else 0 for i in range(n)]
                        total += (-1) ** (len(S) - t_size) * f(assignment)
                if abs(total) > 1e-9:
                    coeffs[S] = total
        return coeffs

    def bits_of(index, n):
        return [(index >> q) & 1 for q in range(n)]

    def spectrum(polynomial, n):
        """Energies indexed by integer basis state (qubit q = bit q of index)."""
        energies = np.zeros(2**n)
        for index in range(2**n):
            energies[index] = evaluate(polynomial, bits_of(index, n))
        return energies

    def poly_to_str(polynomial, names):
        return ", ".join(
            f"{''.join(names[i] for i in sorted(S)) or '1'}: {v:g}"
            for S, v in sorted(
                polynomial.items(), key=lambda kv: (len(kv[0]), sorted(kv[0]))
            )
        )

    return bits_of, evaluate, mobius_coefficients, poly_to_str, spectrum


@app.cell
def _():
    var_names = {0: "x", 1: "y", 2: "z", 3: "w"}
    return (var_names,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: Implement the AND gadget

    Rosenberg's AND penalty is

    $$P_\wedge(x, y, z) \;=\; xy - 2z(x + y) + 3z,$$

    zero exactly when $z = x \wedge y$. **Fill in `and_penalty` below** so it
    returns this as a `{frozenset({vars}): coeff}` dict, the same shape
    `evaluate` and `mobius_coefficients` expect (see §0, and the reference
    notebook's §2 for the fully worked version).

    One thing worth noticing once you're done: the gap between the four
    invalid rows is **not** uniform ($1, 1, 1, 3$, not $1, 1, 1, 1$). That
    specific non-uniform gap is exactly what makes $P_\wedge$ collapse to
    degree 2, a truth table with a *uniform* gap of $1$ on every invalid row
    needs a cubic term instead (try it: feed such a truth table into
    `mobius_coefficients` and look at the degree of what comes back). This is
    why the exercise hands you the formula rather than asking you to search
    for one: not every valid-gap truth table is quadratic, only the right one
    is.
    """)
    return


@app.function
def and_penalty(x, y, z):
    """P = xy - 2z(x+y) + 3z ; zero iff z = x AND y.

    TODO: return the coefficient dict, e.g.
    {frozenset({x, y}): ..., frozenset({x, z}): ..., ...}
    """
    raise NotImplementedError("Exercise 1: implement and_penalty(x, y, z)")


@app.cell(hide_code=True)
def _(mo):
    ex1_check = mo.ui.run_button(label="✅ Check Exercise 1")
    ex1_check
    return (ex1_check,)


@app.cell(hide_code=True)
def _(evaluate, ex1_check, itertools, mo, mobius_coefficients):
    mo.stop(
        not ex1_check.value,
        mo.md("*Fill in `and_penalty` above, then click the button.*"),
    )

    def _run():
        poly = and_penalty(0, 1, 2)
        rows = ["| x | y | z | penalty | valid row? |", "|---|---|---|---|---|"]
        table_ok = True
        for bits in itertools.product([0, 1], repeat=3):
            val = evaluate(poly, bits)
            valid = bits[2] == (bits[0] and bits[1])
            row_ok = (valid and abs(val) < 1e-9) or (not valid and val > 1e-9)
            table_ok = table_ok and row_ok
            rows.append(
                f"| {bits[0]} | {bits[1]} | {bits[2]} | {val:g} | "
                f"{'✅' if row_ok else '❌'} |"
            )
        recovered = mobius_coefficients(lambda b: evaluate(poly, b), 3)
        self_consistent = recovered == poly
        verdict = (
            "**All checks passed ✅** — zero on every valid row, strictly "
            "positive on every invalid row, and feeding the truth table back "
            "through `mobius_coefficients` reproduces your own dict exactly "
            "(the coefficients are forced, not chosen)."
            if table_ok and self_consistent
            else "**Something's off ❌** — check the rows marked ❌ above, or "
            "the coefficients below."
        )
        return mo.vstack(
            [
                mo.md("\n".join(rows)),
                mo.md(f"Möbius self-consistency: {'✅' if self_consistent else '❌'}"),
                mo.md(verdict),
            ]
        )

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
def _(ex1_solution, mo, poly_to_str, var_names):
    mo.stop(not ex1_solution.value, mo.md("*Click to reveal.*"))

    _and_reference = {
        frozenset({0, 1}): 1.0,
        frozenset({0, 2}): -2.0,
        frozenset({1, 2}): -2.0,
        frozenset({2}): 3.0,
    }
    mo.md(f"""
    ```python
    def and_penalty(x, y, z):
        return {{
            frozenset({{x, y}}): 1.0,
            frozenset({{x, z}}): -2.0,
            frozenset({{y, z}}): -2.0,
            frozenset({{z}}): 3.0,
        }}
    ```

    As a dict: `{{{poly_to_str(_and_reference, var_names)}}}`. This is the only
    quadratic polynomial with this truth table any other coefficients
    either don't vanish on the valid rows or don't stay quadratic.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 2: Build OR by reusing AND, not from scratch

    A tempting first move for OR ($z = x \vee y$) is to repeat Exercise 1's
    recipe: write a truth table with penalty $0$ on the four valid rows and
    $1$ on the four invalid ones, then run it through `mobius_coefficients`.
    Try it below, it's given, not an exercise, precisely so you can see it
    fail:
    """)
    return


@app.cell
def _(mobius_coefficients):
    def naive_or_truth(bits):
        x, y, z = bits
        return 0.0 if z == (x or y) else 1.0

    naive_or_polynomial = mobius_coefficients(naive_or_truth, 3)
    naive_or_max_degree = max(len(term) for term in naive_or_polynomial)
    return naive_or_max_degree, naive_or_polynomial


@app.cell(hide_code=True)
def _(mo, naive_or_max_degree, naive_or_polynomial, poly_to_str, var_names):
    mo.md(f"""
    `{{{poly_to_str(naive_or_polynomial, var_names)}}}` — max degree
    **{naive_or_max_degree}**. A cubic `xyz` term, not usable as a two-body
    Ising interaction. The uniform gap was the wrong choice, same as it would
    have been for AND (§ Exercise 1's aside).

    **The fix: reuse the AND gadget you already trust, via De Morgan's law.**
    $z = x \\vee y$ is logically the same statement as
    $(1-z) = (1-x) \\wedge (1-y)$, i.e. $\\neg z = \\neg x \\wedge \\neg y$. So
    instead of inventing a new truth table, evaluate the *known-quadratic*
    AND penalty at the complemented inputs:

    $$P_\\vee(x, y, z) := P_\\wedge(1-x,\\, 1-y,\\, 1-z).$$

    Substituting $x_i \\mapsto 1 - x_i$ is affine — it can't raise degree — so
    whatever degree $P_\\wedge$ has, $P_\\vee$ inherits, in this case staying
    at 2. **Implement `or_truth(bits)` below** as exactly that substitution
    (use the provided `and_reference` gadget so this exercise doesn't depend
    on Exercise 1 being finished), then the check cell runs it through
    `mobius_coefficients` for you, the same recipe as always, just aimed at a
    smarter truth table.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *`and_reference` below is the same AND gadget from Exercise 1, kept
    internally so Exercise 2 onward always has a working AND gadget to build
    on regardless of your progress above.*
    """)
    return


@app.cell
def _():
    and_reference = {
        frozenset({0, 1}): 1.0,
        frozenset({0, 2}): -2.0,
        frozenset({1, 2}): -2.0,
        frozenset({2}): 3.0,
    }
    return (and_reference,)


@app.function
def or_truth(bits):
    """TODO: return evaluate(and_reference, [1 - b for b in bits])."""
    raise NotImplementedError("Exercise 2: implement or_truth(bits)")


@app.cell(hide_code=True)
def _(mo):
    ex2_check = mo.ui.run_button(label="✅ Check Exercise 2")
    ex2_check
    return (ex2_check,)


@app.cell(hide_code=True)
def _(evaluate, ex2_check, itertools, mo, mobius_coefficients):
    mo.stop(
        not ex2_check.value,
        mo.md("*Fill in `or_truth` above, then click the button.*"),
    )

    def _run():
        poly = mobius_coefficients(or_truth, 3)
        max_degree = max(len(term) for term in poly)
        rows = ["| x | y | z | penalty | valid row? |", "|---|---|---|---|---|"]
        table_ok = True
        for bits in itertools.product([0, 1], repeat=3):
            val = evaluate(poly, bits)
            valid = bits[2] == (bits[0] or bits[1])
            row_ok = (valid and abs(val) < 1e-9) or (not valid and val > 1e-9)
            table_ok = table_ok and row_ok
            rows.append(
                f"| {bits[0]} | {bits[1]} | {bits[2]} | {val:g} | "
                f"{'✅' if row_ok else '❌'} |"
            )
        degree_ok = max_degree <= 2
        verdict = (
            f"**All checks passed ✅** — correct truth table, degree "
            f"{max_degree} (quadratic)."
            if table_ok and degree_ok
            else f"**Something's off ❌** — truth table "
            f"{'✅' if table_ok else '❌'}, degree {max_degree} "
            f"{'✅' if degree_ok else '(too high) ❌'}."
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
    ex2_solution = mo.ui.run_button(label="💡 Show solution 2")
    ex2_solution
    return (ex2_solution,)


@app.cell(hide_code=True)
def _(ex2_solution, mo, or_reference, poly_to_str, var_names):
    mo.stop(not ex2_solution.value, mo.md("*Click to reveal.*"))
    mo.md(f"""
    ```python
    def or_truth(bits):
        return evaluate(and_reference, [1 - b for b in bits])

    or_penalty = mobius_coefficients(or_truth, 3)
    ```

    Result: `{{{poly_to_str(or_reference, var_names)}}}` — degree 2, as
    promised. Written out:

    $$P_\\vee(x, y, z) = x + y + z + xy - 2xz - 2yz.$$

    Same trick generalizes: NAND and NOR fall out of AND/OR by negating the
    output variable ($z \\mapsto 1-z$) instead of the inputs.
    """)
    return


@app.cell
def _(and_reference, evaluate, mobius_coefficients):
    def _or_truth_ref(bits):
        return evaluate(and_reference, [1 - b for b in bits])

    or_reference = mobius_coefficients(_or_truth_ref, 3)
    return (or_reference,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 3: QUBO → Ising: the substitution operator

    Substituting $x_i \mapsto \tfrac{1}{2}(I - Z_i)$ turns a $0/1$ variable
    into a qubit operator: $x_i = 0$ becomes the $Z_i = +1$ eigenstate,
    $x_i = 1$ becomes $Z_i = -1$. **Implement `x_as_pauli(var, n)`** so it
    returns that single-qubit substitution as an $n$-qubit `SparsePauliOp`,
    for variable index `var` among `n` total. Recall Qiskit's Pauli labels
    are big-endian (leftmost character = highest qubit index), so variable
    `var` sits at string position `n - 1 - var`.

    $$x_{\text{var}} \;\longmapsto\; \tfrac12 I^{\otimes n} \;-\; \tfrac12\, Z_{\text{var}}$$

    Once this piece is right, assembling a whole polynomial is just: for each
    term, multiply together the substitution operators of every variable in
    it, scale by the coefficient, and sum over terms, that assembler
    (`qubo_to_hamiltonian`) is given below so you can focus on this one
    substitution.
    """)
    return


@app.function
def x_as_pauli(var, n):
    """TODO: return the SparsePauliOp for (I - Z_var) / 2 on n qubits."""
    raise NotImplementedError("Exercise 3: implement x_as_pauli(var, n)")


@app.cell(hide_code=True)
def _(mo):
    ex3_check = mo.ui.run_button(label="✅ Check Exercise 3")
    ex3_check
    return (ex3_check,)


@app.cell(hide_code=True)
def _(SparsePauliOp, ex3_check, mo):
    mo.stop(
        not ex3_check.value,
        mo.md("*Fill in `x_as_pauli` above, then click the button.*"),
    )

    def _reference(var, n):
        label = ["I"] * n
        label[n - 1 - var] = "Z"
        return SparsePauliOp.from_list([("I" * n, 0.5), ("".join(label), -0.5)])

    def _run():
        cases = [(0, 1), (0, 3), (1, 3), (2, 3), (1, 4)]
        rows = ["| var | n | matches reference |", "|---|---|---|"]
        all_ok = True
        for var, n in cases:
            ok = x_as_pauli(var, n).equiv(_reference(var, n))
            all_ok = all_ok and ok
            rows.append(f"| {var} | {n} | {'✅' if ok else '❌'} |")
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
    def x_as_pauli(var, n):
        label = ["I"] * n
        label[n - 1 - var] = "Z"
        return SparsePauliOp.from_list([
            ("I" * n, 0.5),
            ("".join(label), -0.5),
        ])
    ```

    Two terms: the identity contributes $+\tfrac12$, and a single `Z` at
    position `var` contributes $-\tfrac12$, together giving
    $\tfrac12(I - Z_{\text{var}})$.
    """)
    return


@app.cell
def _(SparsePauliOp):
    def qubo_to_hamiltonian(polynomial, n):
        """Map a multilinear 0-1 polynomial to a diagonal SparsePauliOp.

        Given so Exercise 3 stays focused on the substitution itself; uses
        its own internal copy of that substitution so it runs regardless of
        whether x_as_pauli above is finished yet.
        """

        def _x_as_pauli(var, n):
            label = ["I"] * n
            label[n - 1 - var] = "Z"
            return SparsePauliOp.from_list([("I" * n, 0.5), ("".join(label), -0.5)])

        identity = SparsePauliOp.from_list([("I" * n, 1.0)])
        hamiltonian = SparsePauliOp.from_list([("I" * n, 0.0)])
        for term, coeff in polynomial.items():
            factor = identity * coeff
            for var in term:
                factor = factor.compose(_x_as_pauli(var, n))
            hamiltonian = hamiltonian + factor
        return hamiltonian.simplify()

    return (qubo_to_hamiltonian,)


@app.cell
def _(mo, or_reference, qubo_to_hamiltonian):
    h_or = qubo_to_hamiltonian(or_reference, 3)

    _lines = ["| Pauli term | coefficient |", "|---|---|"]
    for _pauli, _coeff in sorted(
        zip(h_or.paulis.to_labels(), h_or.coeffs), key=lambda kv: kv[0]
    ):
        _lines.append(f"| `{_pauli}` | {_coeff.real:+.3f} |")
    mo.md("**OR penalty as an Ising Hamiltonian:**\n\n" + "\n".join(_lines))
    return (h_or,)


@app.cell(hide_code=True)
def _(h_or, mo, np, or_reference, spectrum):
    _diag = np.real(np.diag(h_or.to_matrix()))
    _bf = spectrum(or_reference, 3)
    _match = np.allclose(_diag, _bf)
    mo.md(
        f"Brute-force polynomial vs. `SparsePauliOp` diagonal: "
        f"**{'match ✅' if _match else 'MISMATCH ❌'}** "
        f"(max deviation {np.max(np.abs(_diag - _bf)):.2e}), the substitution "
        "step didn't change what's being computed, only its representation."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 4: Penalty weight vs. a competing objective

    Take the objective $f = -x - y + 2z$. It wants $x = y = 1$ (cheap) while
    also wanting $z = 0$ (cheap), a direct contradiction of the OR
    constraint, since $x=y=1$ forces $z=1$. The full Hamiltonian is

    $$H = f + \lambda \, P_\vee(x, y, z),$$

    and the ground state is only guaranteed feasible once $\lambda$ is large
    enough to outweigh what the objective gains by cheating.

    **Implement `combine(objective, penalty, lam)`**: merge two coefficient
    dicts into `objective + lam * penalty`, term by term. Then use the slider
    below to search for the crossover $\lambda$ by eye, before revealing the
    exact value.
    """)
    return


@app.cell
def _():
    objective = {frozenset({0}): -1.0, frozenset({1}): -1.0, frozenset({2}): 2.0}
    return (objective,)


@app.function
def combine(objective, penalty, lam):
    """TODO: return {term: objective.get(term, 0) + lam * penalty.get(term, 0)
    for every term appearing in either dict}."""
    raise NotImplementedError("Exercise 4: implement combine(objective, penalty, lam)")


@app.cell(hide_code=True)
def _(mo):
    ex4_check = mo.ui.run_button(label="✅ Check Exercise 4")
    ex4_check
    return (ex4_check,)


@app.cell(hide_code=True)
def _(evaluate, ex4_check, itertools, mo, objective, or_reference):
    mo.stop(
        not ex4_check.value,
        mo.md("*Fill in `combine` above, then click the button.*"),
    )

    def _reference(objective, penalty, lam):
        total = dict(objective)
        for term, coeff in penalty.items():
            total[term] = total.get(term, 0.0) + lam * coeff
        return total

    def _run():
        all_ok = True
        rows = ["| λ | max |energy diff| over all bitstrings | match |", "|---|---|---|"]
        for lam in (0.0, 0.7, 1.5):
            got = combine(objective, or_reference, lam)
            want = _reference(objective, or_reference, lam)
            diffs = [
                abs(evaluate(got, bits) - evaluate(want, bits))
                for bits in itertools.product([0, 1], repeat=3)
            ]
            ok = max(diffs) < 1e-9
            all_ok = all_ok and ok
            rows.append(f"| {lam:.1f} | {max(diffs):.2e} | {'✅' if ok else '❌'} |")
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
    def combine(objective, penalty, lam):
        total = dict(objective)
        for term, coeff in penalty.items():
            total[term] = total.get(term, 0.0) + lam * coeff
        return total
    ```

    Nothing fancier than a merge over the union of both dicts' keys, using
    `dict.get(term, 0.0)` so terms present in only one polynomial don't raise
    a `KeyError`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lam_slider = mo.ui.slider(
        0.0, 2.0, value=0.3, step=0.05, label="penalty weight λ", show_value=True
    )
    lam_slider
    return (lam_slider,)


@app.cell(hide_code=True)
def _(bits_of, lam_slider, np, objective, or_reference, plt, spectrum):
    def _combine(objective, penalty, lam):
        total = dict(objective)
        for term, coeff in penalty.items():
            total[term] = total.get(term, 0.0) + lam * coeff
        return total

    _total = _combine(objective, or_reference, lam_slider.value)
    _energies = spectrum(_total, 3)
    _ground = int(np.argmin(_energies))
    _labels = ["".join(str(b) for b in bits_of(_i, 3)) for _i in range(8)]
    _valid = [
        bits_of(_i, 3)[2] == (bits_of(_i, 3)[0] | bits_of(_i, 3)[1])
        for _i in range(8)
    ]

    fig_lambda, _ax = plt.subplots(figsize=(7, 3.4))
    _colors = ["#55a868" if v else "#c44e52" for v in _valid]
    _bars = _ax.bar(_labels, _energies, color=_colors)
    _bars[_ground].set_edgecolor("black")
    _bars[_ground].set_linewidth(2.5)
    _ax.axhline(0, color="k", lw=0.8)
    _ax.set_xlabel("assignment  (x y z)")
    _ax.set_ylabel("energy")
    _ax.set_title(
        f"λ = {lam_slider.value:.2f} → ground state {_labels[_ground]} "
        f"({'feasible' if _valid[_ground] else 'INFEASIBLE'}), outlined in black"
    )
    fig_lambda.tight_layout()
    fig_lambda
    return


@app.cell(hide_code=True)
def _(mo):
    ex4_threshold = mo.ui.run_button(label="💡 Reveal the exact threshold")
    ex4_threshold
    return (ex4_threshold,)


@app.cell(hide_code=True)
def _(ex4_threshold, mo):
    mo.stop(not ex4_threshold.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    The crossover is at **$\lambda^\* = 1$**. Below it, the minimiser is the
    infeasible `x=0, y=1, z=0` (or `x=1,y=0,z=0`, tied), which nets
    $f = -1$ while paying only `λ` in penalty; above $\lambda = 1$ the
    penalty finally outweighs that saving and a feasible assignment
    (`0 0 0` or `1 1 1`, degenerate at energy $0$) takes over. Same story as
    the reference notebook's AND example (§5): too small a $\lambda$ silently
    returns an infeasible answer, too large compresses the objective's energy
    scale for no further benefit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 5: Capstone: solve for the bitstring with Qiskit itself

    Every check so far read the answer off a classical array (`spectrum` /
    `.to_matrix()`). This exercise does it the way a variational algorithm
    would: **build a `QuantumCircuit` for each candidate bitstring, and use a
    Qiskit primitive to estimate its energy.**

    **Implement `solve_via_qiskit(hamiltonian, n)`**: for each of the $2^n$
    computational basis bitstrings, build an $n$-qubit `QuantumCircuit` that
    prepares it (an `x` gate on every qubit that should be $1$, nothing
    otherwise), estimate $\langle \psi_{\text{bits}} | H | \psi_{\text{bits}}
    \rangle$ with `StatevectorEstimator`, and return `(best_bitstrings,
    energies)` where `best_bitstrings` is the list of bitstrings tying for
    minimum energy and `energies` is a `{bitstring_tuple: energy}` dict for
    all $2^n$ of them.

    `StatevectorEstimator().run([(circuit, hamiltonian), ...]).result()`
    accepts a batch of `(circuit, operator)` pairs at once and returns one
    result per pair, in order; each result's expectation value is at
    `result[i].data.evs`.
    """)
    return


@app.function
def solve_via_qiskit(hamiltonian, n):
    """TODO: return (best_bitstrings, energies) as described above."""
    raise NotImplementedError(
        "Exercise 5: implement solve_via_qiskit(hamiltonian, n)"
    )


@app.cell(hide_code=True)
def _(mo):
    ex5_check = mo.ui.run_button(label="✅ Check Exercise 5")
    ex5_check
    return (ex5_check,)


@app.cell(hide_code=True)
def _(bits_of, ex5_check, h_or, mo, or_reference, spectrum):
    mo.stop(
        not ex5_check.value,
        mo.md("*Fill in `solve_via_qiskit` above, then click the button.*"),
    )

    def _run():
        best, energies = solve_via_qiskit(h_or, 3)
        bf_energies = spectrum(or_reference, 3)
        rows = ["| bitstring | your energy | brute-force energy | match |", "|---|---|---|---|"]
        all_ok = True
        for _i in range(8):
            bits = tuple(bits_of(_i, 3))
            got = energies.get(bits)
            want = bf_energies[_i]
            ok = got is not None and abs(got - want) < 1e-8
            all_ok = all_ok and ok
            rows.append(
                f"| {''.join(map(str, bits))} | "
                f"{'—' if got is None else f'{got:.3f}'} | {want:.3f} | "
                f"{'✅' if ok else '❌'} |"
            )
        best_ok = set(best) == {
            tuple(bits_of(_i, 3))
            for _i in range(8)
            if abs(bf_energies[_i] - bf_energies.min()) < 1e-9
        }
        all_ok = all_ok and best_ok
        verdict = (
            f"**All checks passed ✅** — best bitstring(s): "
            f"{[''.join(map(str, b)) for b in best]}."
            if all_ok
            else "**Something's off ❌** — check the rows above, and whether "
            "`best_bitstrings` matches the true minimisers."
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
    ex5_solution = mo.ui.run_button(label="💡 Show solution 5")
    ex5_solution
    return (ex5_solution,)


@app.cell(hide_code=True)
def _(ex5_solution, mo):
    mo.stop(not ex5_solution.value, mo.md("*Click to reveal.*"))
    mo.md(r"""
    ```python
    def solve_via_qiskit(hamiltonian, n):
        estimator = StatevectorEstimator()
        pubs, bitstrings = [], []
        for index in range(2 ** n):
            bits = [(index >> q) & 1 for q in range(n)]
            qc = QuantumCircuit(n)
            for q, b in enumerate(bits):
                if b:
                    qc.x(q)
            pubs.append((qc, hamiltonian))
            bitstrings.append(tuple(bits))

        result = estimator.run(pubs).result()
        energies = {bs: float(r.data.evs) for bs, r in zip(bitstrings, result)}

        best_energy = min(energies.values())
        best_bitstrings = [
            bs for bs, e in energies.items() if abs(e - best_energy) < 1e-9
        ]
        return best_bitstrings, energies
    ```

    Preparing $|x\rangle$ with `x` gates and measuring $\langle x|H|x\rangle$
    is the trivial case of what QAOA/VQE do with a *parameterised* circuit
    instead of a fixed bitstring — this exercise is the $t=0$ version of that
    workflow, using exactly the Qiskit primitives interface those algorithms
    use.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Capstone run

    Your `solve_via_qiskit` from Exercise 5, pointed at the harder Exercise 4
    Hamiltonian ($f + 1.5\, P_\vee$, safely past the $\lambda = 1$ feasibility
    threshold): every bitstring gets its own circuit and its own primitive
    call, and the plot below is built entirely from what came back.
    """)
    return


@app.cell(hide_code=True)
def _(bits_of, mo, objective, or_reference, plt, qubo_to_hamiltonian):
    def _combine(objective, penalty, lam):
        total = dict(objective)
        for term, coeff in penalty.items():
            total[term] = total.get(term, 0.0) + lam * coeff
        return total

    def _run():
        h_capstone = qubo_to_hamiltonian(_combine(objective, or_reference, 1.5), 3)
        best, energies = solve_via_qiskit(h_capstone, 3)

        _labels = ["".join(str(b) for b in bits_of(_i, 3)) for _i in range(8)]
        _values = [energies[tuple(bits_of(_i, 3))] for _i in range(8)]
        _is_best = [tuple(bits_of(_i, 3)) in best for _i in range(8)]

        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.bar(_labels, _values, color=["#55a868" if b else "#4c72b0" for b in _is_best])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("assignment  (x y z)")
        ax.set_ylabel("energy (from StatevectorEstimator)")
        ax.set_title(
            f"Winner(s): {[''.join(map(str, b)) for b in best]}, "
            "found by simulating every bitstring with Qiskit"
        )
        fig.tight_layout()
        return fig

    try:
        _result = _run()
    except NotImplementedError:
        _result = mo.md(
            "*Finish Exercise 5's `solve_via_qiskit` above to see this run.*"
        )
    _result
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where to go from here

    You've rebuilt the QUBO → Ising pipeline: a hand-implemented gadget
    (AND), a reused/derived one (OR via De Morgan), the $x_i \mapsto
    \tfrac12(I - Z_i)$ substitution, a penalty-weight tradeoff, and a
    Qiskit-primitives solve of "which bitstring wins."

    The reference notebook `boolean_penalty_hamiltonians.py` goes one step
    further into **degree reduction** (§6): building XOR, whose natural
    penalty is cubic, by introducing an ancilla qubit constrained with the
    same AND gadget from Exercise 1. Worth reading once these five feel
    solid. It also points to `qiskit-optimization`'s `QuadraticProgram` /
    `QuadraticProgramToQubo` (§7), which automates everything done here by
    hand, including penalty-weight selection doing it manually once, as
    above, is what makes it possible to debug that automation when its
    default weight turns out to be wrong for your problem.
    """)
    return


if __name__ == "__main__":
    app.run()
