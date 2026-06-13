# /// script
# dependencies = [
#     "marimo",
#     "matplotlib==3.11.0",
#     "pylatexenc==2.10",
#     "qiskit==2.4.2",
#     "qiskit-aer==0.17.2",
#     "qiskit-algorithms==0.4.0",
#     "qiskit-machine-learning==0.9.0",
#     "scikit-learn==1.9.0",
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
    ## Quantum Machine Learning

    QML can be understood in different way but here we will focus on how we can take classical data and use quantum circuits to learn from such data in order to perform classification tasks. The field is much more than this but getting these basic concepts will help you set a solid foundation moving forward.

    ### Data embedding

    Quantum computers handle quantum states. So, what do I do if I only have classical data? Quantum embedding deals with this, trying to find the quantum representation of your data points. While doing this, if we can actually improve their representation so that data becomes more separable, the better.
    """)
    return


@app.cell
def _():
    from qiskit.circuit.library import zz_feature_map

    feature_map = zz_feature_map(3, reps=1)
    feature_map.decompose().draw('mpl', fold=-1)
    return feature_map, zz_feature_map


@app.cell
def _(zz_feature_map):
    feature_map_2 = zz_feature_map(3, reps=2, entanglement="full")
    feature_map_2.draw('mpl', fold=-1)
    return


@app.cell
def _():
    from qiskit.circuit.library import pauli_feature_map

    feature_map_pauli = pauli_feature_map(3, paulis=["X", "YZ"], entanglement='linear')
    feature_map_pauli.decompose().draw('mpl', fold=-1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So if you have a dataset with three features (columns), it is straightforward to produce quantum states using those three values as angle rotations using this quantum embedding.
    """)
    return


@app.cell
def _(feature_map):
    from qiskit import QuantumCircuit

    qcirc = QuantumCircuit(3,3)
    qcirc.compose(feature_map, inplace=True)
    qcirc.measure([0,1,2], [0,1,2])

    qcirc.draw(fold=-1, style="mpl")
    return QuantumCircuit, qcirc


@app.cell
def _(feature_map):
    feature_map.parameters
    return


@app.cell
def _(qcirc):
    circ = qcirc.assign_parameters({
        'x[0]' : 0.01,
        'x[1]' : 0.5,
        'x[2]' : 0.9,
    })
    return (circ,)


@app.cell
def _(circ):
    from qiskit_aer import AerSimulator

    # execute the quantum circuit
    simulator = AerSimulator()

    result = simulator.run(circ, shots=1000).result()
    counts  = result.get_counts(circ)
    print(counts)
    return (counts,)


@app.cell
def _(counts):
    from qiskit.visualization import plot_histogram

    plot_histogram(counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Where do we get those parameter values from? Let's try with some sample datasets from the [Scikit Learn library](https://scikit-learn.org/stable/).
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn import datasets
    from sklearn.decomposition import PCA

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:100, :2]  # we only take the first two features.
    Y = iris.target[:100] # and first 100 samples

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xticks(())
    plt.yticks(())

    plt.show()
    return X, Y, iris, plt


@app.cell
def _(X, Y):
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.33,
        random_state=42,
    )

    scaler = MinMaxScaler((-0.9, 0.9))

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return MinMaxScaler, X_test, X_train, train_test_split, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We create a dimension appropriate feature map.
    """)
    return


@app.cell
def _(X_train):
    from qiskit.circuit.library import z_feature_map

    fmap = z_feature_map(X_train.shape[1], reps=1)
    fmap.decompose().draw('mpl')
    return fmap, z_feature_map


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Kernels
    """)
    return


@app.cell
def _(X_train, fmap):
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_algorithms.state_fidelities import ComputeUncompute
    from qiskit_aer.primitives import SamplerV2 as Sampler

    # How to compute the fidelity between the states 
    fidelity = ComputeUncompute(sampler=Sampler())

    # Feature map and quantum Kernel
    kernel = FidelityQuantumKernel(feature_map=fmap, fidelity=fidelity)
    kernel.evaluate(X_train[0,:], X_train[0,:])
    return FidelityQuantumKernel, fidelity, kernel


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What is this ComputeUncompute class? It creates when in machine learning is know as the **kernel trick**. This trick allos us to use a function as a measure of distance in a higher dimensional state. So, given two samples grom our dataset we could obtain

    $$
    K(x_i, x_j) \ge 0
    $$
    which for the same sample should return $K(x_i,x_i) = 0$ distance as they are the same point in the kernel dimensional space.

    A quantum circuit can be viewed as the kernel function operating in a similar manner:

    $$
    K(x_i, x_j) = \langle\psi(x_i)|\psi(x_j)\rangle = \langle 0 |\Psi^{\dagger}(x)|\Psi(x_j)|0\rangle
    $$
    so that $\Psi$ is the function describing our quantum cirucit and its conjugate pefforms the operation backwards. This produces a **compute - uncompute** dynamic returning $|0\rangle$ state when operations being performed are the same in parameterized actions in both cases. When measuring fidelity as outcome, this scale is inverted 1 meaning a complete overlap for the expected outcome $|0\rangle$.

    This is the key to [Quantum Support Vector Classifiers](https://iraitzm.github.io/qc-handbook/parts/qml/kernels.html#quantum-kernel)
    """)
    return


@app.cell
def _(X_train, kernel):
    kernel.evaluate(X_train[1,:], X_train[0,:])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Thus, we know that samples 0 and 1 from our dataset produce quantum states that overlap just 0.35, meaning they are quite ditant in that quantum feature space without the need of observing the actual states they produce.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantum Support Vector Classifier

    We now can use this concept of distance trying to obatin a classical model that can create hyperplanes so that this data becomes more separable.
    """)
    return


@app.cell
def _(X_test, X_train, kernel, y_test, y_train):
    from qiskit_machine_learning.algorithms import QSVC

    # QKernel + SVC
    qsvc = QSVC(quantum_kernel=kernel)
    qsvc.fit(X_train, y_train)
    qsvc.score(X_test, y_test)
    return (QSVC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Perfect score, meaning our model can perfectly distinguish between 0 and 1 labeled samples. Well, the original information was already separable enough, don't you think? Let's try witha  harder example.
    """)
    return


@app.cell
def _(iris, plt):
    X_hard = iris.data[50:150, :2]  # we only take the last two features.
    Y_hard = iris.target[50:] # and first 100 samples

    plt.figure(2, figsize=(8, 6))
    plt.clf()

    # Plot the training points
    plt.scatter(X_hard[:, 0], X_hard[:, 1], c=Y_hard, cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xticks(())
    plt.yticks(())

    plt.show()
    return X_hard, Y_hard


@app.cell
def _(
    FidelityQuantumKernel,
    MinMaxScaler,
    QSVC,
    X_hard,
    Y_hard,
    fidelity,
    train_test_split,
    z_feature_map,
):
    # Train and test
    X_train_hard, X_test_hard, y_train_hard, y_test_hard = train_test_split(
        X_hard,
        Y_hard,
        test_size=0.33,
        random_state=42,
    )

    # Scale
    _scaler = MinMaxScaler((-0.9, 0.9))
    X_train_scl = _scaler.fit_transform(X_train_hard)
    X_test_scl = _scaler.transform(X_test_hard)

    # Feature map and kernel composition
    fmap_h = z_feature_map(X_train_scl.shape[1], reps=1)

    # Feature map and quantum Kernel
    kernel_h = FidelityQuantumKernel(feature_map=fmap_h, fidelity=fidelity)

    # QKernel + SVC
    qsvc_h = QSVC(quantum_kernel=kernel_h)
    qsvc_h.fit(X_train_scl, y_train_hard)
    qsvc_h.score(X_test_scl, y_test_hard)
    return X_test_scl, X_train_scl, qsvc_h, y_test_hard, y_train_hard


@app.cell
def _(X_test_scl, qsvc_h, y_test_hard):
    from sklearn.metrics import classification_report

    y_pred = qsvc_h.predict(X_test_scl)
    print(classification_report(y_test_hard, y_pred))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise

    Can you propose a different feature map (ZZ maybe?) and check if results improve?
    """)
    return


@app.cell
def _():
    # YOUR CODE GOES HERE
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quantum Neural Networks

    QNN's take it one step further, considering we already embbed the data and maybe we could define something like an open anstaz and find the right set of parameters so that the circuit itself produces a classification over the sample being encoded. This removes the need to compare a given sample against a set of support samples, reducing the amount of execution during inference.
    """)
    return


@app.cell
def _(QuantumCircuit, zz_feature_map):
    from qiskit.circuit.library import real_amplitudes

    num_inputs = 2
    feature_map_nn = zz_feature_map(num_inputs)
    ansatz = real_amplitudes(num_inputs, reps=1)

    # construct QNN
    qnn = QuantumCircuit(num_inputs)
    qnn.compose(feature_map_nn, inplace=True)
    qnn.compose(ansatz, inplace=True)
    qnn.draw(output="mpl")
    return ansatz, feature_map_nn, qnn


@app.cell
def _(
    X_test_scl,
    X_train_scl,
    ansatz,
    feature_map_nn,
    qnn,
    y_test_hard,
    y_train_hard,
):
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
    from qiskit_machine_learning.neural_networks import EstimatorQNN

    # We append the estructure to the Estimator primitive
    estimator_qnn = EstimatorQNN(
        circuit=qnn, input_params=feature_map_nn.parameters, weight_params=ansatz.parameters
    )

    # construct neural network classifier
    estimator_classifier = NeuralNetworkClassifier(
        estimator_qnn, optimizer=COBYLA(maxiter=60)
    )

    estimator_classifier.fit(X_train_scl, y_train_hard)
    estimator_classifier.score(X_test_scl, y_test_hard)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we need to get creative on defining both the feature map but also the ansatz that will allow us to improve that metric.
    """)
    return


if __name__ == "__main__":
    app.run()
