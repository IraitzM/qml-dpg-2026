# /// script
# dependencies = [
#     "marimo",
#     "numpy==2.4.6",
#     "quimb==1.14.0",
#     "scikit-learn==1.9.0",
#     "torch==2.12.1",
#     "torchvision==0.27.1",
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
    Neural networks have captured the media by storm in particular since the irruption of Large Language Models, humongous neural networks able to autorregresively propose the most likely word in a sentence.

    [Detailes view](https://bbycroft.net/llm)

    But, in essence this are nothing but matrix multiplications where we would need to find the right values for each cell (weights).
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn.functional as F
    import numpy as np
    import quimb.tensor as qtn
    from torchvision import datasets, transforms
    from sklearn.decomposition import PCA

    torch.manual_seed(42)
    np.random.seed(42)
    return F, PCA, datasets, qtn, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will replace our fancy Neural Network by  a TN structure that essentially will perform the same actions: find the set of values that stablish the relationship between entry data and its label. Thus, *learning* the relationship between those.

    Allow us to define a set of parameters for the whole setup.
    """)
    return


@app.cell
def _():
    N_SITES = 20     # MPS sites = number of PCA components kept
    D_PHYS  = 2      # local feature dimension (cos/sin encoding → always 2)
    CHI     = 10     # bond dimension χ  ← the main knob to tune
    N_CLASS = 10     # one output per digit class
    BATCH   = 256
    EPOCHS  = 10
    LR      = 3e-3
    N_TRAIN = 12_000  # subset of MNIST training set
    N_TEST  = 2_000   # subset of MNIST test set
    return BATCH, CHI, D_PHYS, EPOCHS, LR, N_CLASS, N_SITES, N_TEST, N_TRAIN


@app.cell
def _(N_SITES, N_TEST, N_TRAIN, PCA, datasets, torch, transforms):
    def load_and_preprocess():
        """
        Returns X_train, y_train, X_test, y_test as torch tensors.
        Each X has shape (N, N_SITES) with values in [0, 1].
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),   # flatten 28×28 → 784
        ])
        train_ds = datasets.MNIST("./data", train=True,  download=True,  transform=transform)
        test_ds  = datasets.MNIST("./data", train=False, download=True,  transform=transform)

        X_tr = torch.stack([train_ds[i][0] for i in range(N_TRAIN)]).numpy()
        y_tr = torch.tensor([train_ds[i][1] for i in range(N_TRAIN)])
        X_te = torch.stack([test_ds[i][0]  for i in range(N_TEST)]).numpy()
        y_te = torch.tensor([test_ds[i][1] for i in range(N_TEST)])

        # PCA: 784 → N_SITES
        pca = PCA(n_components=N_SITES, random_state=42)
        pca.fit(X_tr)
        X_tr_p = pca.transform(X_tr)
        X_te_p = pca.transform(X_te)

        # Per-component min-max normalization to [0, 1] using training statistics
        mn, mx = X_tr_p.min(0), X_tr_p.max(0)
        X_tr_n = (X_tr_p - mn) / (mx - mn + 1e-8)
        X_te_n = (X_te_p - mn) / (mx - mn + 1e-8)

        var_explained = pca.explained_variance_ratio_.sum() * 100
        print(f"PCA variance explained: {var_explained:.1f}% with {N_SITES} components")

        return (
            torch.tensor(X_tr_n, dtype=torch.float32), y_tr,
            torch.tensor(X_te_n, dtype=torch.float32), y_te,
        )

    return (load_and_preprocess,)


@app.cell
def _(torch):
    def feature_map(x_batch: torch.Tensor) -> torch.Tensor:
        """
        Map each scalar xᵢ ∈ [0, 1] to a local 2-vector:
            φ(xᵢ) = [cos(π xᵢ / 2),  sin(π xᵢ / 2)]

        This is the same encoding used in Stoudenmire & Schwab (2016).
        It guarantees ||φ(xᵢ)||₂ = 1 for all inputs, which prevents the
        chain product from decaying exponentially with site number.

        Args:
            x_batch: (B, N)  — batch of N-component feature vectors

        Returns:
            phi: (B, N, 2)
        """
        return torch.stack(
            [torch.cos(torch.pi * x_batch / 2),
             torch.sin(torch.pi * x_batch / 2)],
            dim=-1
        )

    return (feature_map,)


@app.cell
def _(torch):
    def init_mps_params(N: int, d: int, chi: int, n_class: int) -> torch.nn.ParameterList:
        """
        Build the N trainable tensors of the weight MPS.

        Layout:
            Site 0      :  (d, χ)               — left boundary
            Sites 1..N-2:  (χ, d, χ)            — bulk
            Site N-1    :  (χ, d, n_class)       — right boundary; n_class replaces right bond

        Initialisation strategy (avoids vanishing/exploding norms):
            - Site 0: random, normalised to 1/√χ
            - Bulk:   random rectangular matrix reshaped to (χ, d, χ),
                      left-orthogonalised via SVD, then divided by χ^0.25
            - Last:   small random, scaled by 1/√(χ·d)
        """
        ps = []

        # Left boundary
        W = torch.randn(d, chi)
        W = W / W.norm() / chi ** 0.5
        ps.append(torch.nn.Parameter(W))

        # Bulk sites
        for _ in range(N - 2):
            W = torch.randn(chi * d, chi)
            W, _, _ = torch.linalg.svd(W, full_matrices=False)   # left-ortho columns
            W = (W / chi ** 0.25).reshape(chi, d, chi)
            ps.append(torch.nn.Parameter(W))

        # Right boundary (has class output index instead of right bond)
        W = torch.randn(chi, d, n_class) / (chi * d) ** 0.5
        ps.append(torch.nn.Parameter(W))

        n_params = sum(p.numel() for p in ps)
        print(f"MPS: {N} sites | χ={chi} | d={d} | {n_params:,} parameters")
        return torch.nn.ParameterList(ps)


    return (init_mps_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every NN uses two functions in particular:

    * **Forward pass**: Given a set of inputs, performs the forward pass multiplying by the weights and producing the final outcome, label.
    * **Backward pass**: Defines how far from the ideal label we are and projects this error backwards to correct the weigths and thus learn minimizing the *loss* function.
    """)
    return


@app.cell
def _(torch):
    def mps_forward(
        phi: torch.Tensor,
        params: torch.nn.ParameterList,
        N: int,
    ) -> torch.Tensor:
        """
        Contract the feature map φ into the weight MPS, site by site (left to right).

        At each site i the bond vector is updated as:
            v[b, k] = Σ_{j,p}  v_prev[b, j] · A[i][j, p, k] · φ[b, i, p]

        where A[i] is the weight tensor at site i and φ[b, i, :] is the local
        feature for sample b at site i.

        The intermediate bond vector is renormalised after each site to prevent
        exponential decay of activations through the chain.

        Args:
            phi   : (B, N, d)  — batch of feature vectors
            params: list of N weight tensors
            N     : number of MPS sites

        Returns:
            logits: (B, n_class)
        """
        # Site 0: (B, d) @ (d, χ) → (B, χ)
        v = phi[:, 0, :] @ params[0]
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

        # Bulk sites 1 … N-2
        for i in range(1, N - 1):
            # v: (B, χ),  params[i]: (χ, d, χ),  phi[:,i]: (B, d)
            v = torch.einsum("bj, jpk, bp -> bk", v, params[i], phi[:, i, :])
            v = v / (v.norm(dim=1, keepdim=True) + 1e-8)

        # Last site: (χ, d, n_class) → output (B, n_class)
        logits = torch.einsum("bj, jpk, bp -> bk", v, params[-1], phi[:, N - 1, :])
        return logits

    return (mps_forward,)


@app.cell
def _(qtn, torch):
    def build_quimb_mps(params: torch.nn.ParameterList, N: int) -> qtn.MatrixProductState:
        """
        Wrap the trained weight tensors in a quimb MatrixProductState.
        Useful for inspecting bond dimensions, computing overlaps,
        running SVD compression, or doing DMRG-style sweeps.

        Note: The last site has shape (χ, d, n_class) which is non-standard;
        quimb treats it as a bulk tensor with an extra 'class' index.
        """
        arrays = [p.detach().numpy() for p in params]
        # quimb expects:  site 0: (d, χ),  bulk: (χ, d, χ),  last: (χ, d)
        # Our last site has shape (χ, d, n_class) — pass it as a raw TensorNetwork
        # for inspection instead.
        tn = qtn.TensorNetwork([
            qtn.Tensor(
                arrays[i],
                inds=(
                    [f"p{i}", f"b{i}"]               if i == 0 else
                    [f"b{i-1}", f"p{i}", f"b{i}"]    if i < N - 1 else
                    [f"b{i-1}", f"p{i}", "class"]
                ),
                tags=[f"I{i}", "MPS"]
            )
            for i in range(N)
        ])
        return tn

    return (build_quimb_mps,)


@app.cell
def _(BATCH, EPOCHS, F, LR, N_SITES, N_TRAIN, feature_map, mps_forward, torch):
    def train(params, X_tr, y_tr, X_te, y_te):
        opt   = torch.optim.Adam(params.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

        history = []
        print(f"\n{'Epoch':>5} | {'Loss':>9} | {'Test acc':>8}")
        print("-" * 32)

        for ep in range(EPOCHS):
            perm = torch.randperm(N_TRAIN)
            total_loss, n_b = 0.0, 0

            for i in range(0, N_TRAIN, BATCH):
                idx    = perm[i : i + BATCH]
                phi    = feature_map(X_tr[idx])
                logits = mps_forward(phi, params, N_SITES)
                loss   = F.cross_entropy(logits, y_tr[idx]) # Loss function

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params.parameters(), max_norm=1.0)
                opt.step()
                total_loss += loss.item()
                n_b += 1

            sched.step()

            with torch.no_grad():
                phi_te = feature_map(X_te)
                acc = (mps_forward(phi_te, params, N_SITES).argmax(1) == y_te).float().mean().item()

            avg_loss = total_loss / n_b
            history.append({"epoch": ep + 1, "loss": avg_loss, "acc": acc})
            print(f"{ep+1:5d} | {avg_loss:9.4f} | {acc*100:7.1f}%")

        return history


    return (train,)


@app.cell
def _(load_and_preprocess):
    X_tr, y_tr, X_te, y_te = load_and_preprocess()
    return X_te, X_tr, y_te, y_tr


@app.cell
def _(CHI, D_PHYS, N_CLASS, N_SITES, init_mps_params):
    params = init_mps_params(N_SITES, D_PHYS, CHI, N_CLASS)
    return (params,)


@app.cell
def _(CHI, N_CLASS, N_SITES, build_quimb_mps, params):
    print("\nQuimb tensor network (weight MPS):")
    tn = build_quimb_mps(params, N_SITES)
    print(f"  {N_SITES} tensors, outer indices: {[f'p{i}' for i in range(N_SITES)]} (physical)")
    print(f"  bond indices: {[f'b{i}' for i in range(N_SITES-1)]} (bond dim ={CHI})")
    print(f"  class index : 'class'  (dim={N_CLASS})")
    return


@app.cell
def _(X_te, X_tr, params, train, y_te, y_tr):
    import time

    t0 = time.time()
    history = train(params, X_tr, y_tr, X_te, y_te)
    print(f"\nTraining time: {time.time()-t0:.1f}s")
    print(f"Final test accuracy: {history[-1]['acc']*100:.1f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could also analyze how the bond dimension affects the learning, but more importantly how much we might be willing to sacrifice in favor of computationa, efficiency.
    """)
    return


@app.cell
def _(
    BATCH,
    D_PHYS,
    EPOCHS,
    F,
    LR,
    N_CLASS,
    N_SITES,
    N_TRAIN,
    feature_map,
    init_mps_params,
    mps_forward,
    torch,
):
    def bond_sweep(X_tr, y_tr, X_te, y_te, chi_values=(1, 2, 4, 8, 16)):
        """
        Quick training at each χ for EPOCHS//2 epochs to compare expressivity.
        """
        print(f"{'χ':>4} | {'# params':>8} | {'Test acc':>8}")
        print("-" * 28)
        results = []
        for chi in chi_values:
            ps = init_mps_params(N_SITES, D_PHYS, chi, N_CLASS)
            opt = torch.optim.Adam(ps.parameters(), lr=LR)
            for ep in range(EPOCHS // 2):
                perm = torch.randperm(N_TRAIN)
                for i in range(0, N_TRAIN, BATCH):
                    idx    = perm[i : i + BATCH]
                    phi    = feature_map(X_tr[idx])
                    logits = mps_forward(phi, ps, N_SITES)
                    loss   = F.cross_entropy(logits, y_tr[idx])
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(ps.parameters(), 1.0)
                    opt.step()
            with torch.no_grad():
                phi_te = feature_map(X_te)
                acc = (mps_forward(phi_te, ps, N_SITES).argmax(1) == y_te).float().mean().item()
            n_p = sum(p.numel() for p in ps)
            print(f"{chi:4d} | {n_p:8,} | {acc*100:7.1f}%")
            results.append({"chi": chi, "n_params": n_p, "acc": acc})
        return results

    return (bond_sweep,)


@app.cell
def _(X_te, X_tr, bond_sweep, y_te, y_tr):
    sweep = bond_sweep(X_tr, y_tr, X_te, y_te, chi_values=[1, 2, 4, 8, 16])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Probably this exercise gives you a hint on what is going on for example when companies like Multiverse Computing [claim](https://arxiv.org/pdf/2401.14109) they are able to reduce the weights of billion parameter models to a small portion of those still with good performance.
    """)
    return


if __name__ == "__main__":
    app.run()
