# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.11",
#     "matplotlib==3.11.0",
#     "numpy==2.4.6",
#     "pennylane==0.45.1",
#     "pennylane-lightning==0.45.0",
#     "pennylane-lightning-gpu==0.45.0",
#     "pennylane-lightning-tensor==0.45.0",
#     "pylatexenc==2.10",
#     "quimb==1.14.0",
#     "scikit-learn==1.9.0",
#     "squlearn==0.11.2",
# ]
# ///

import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    import warnings
    warnings.filterwarnings('ignore')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Can we summarize all the previous effort so that we can benefit from TN performance when training QML models as well? Short answer, yes. And even shorter code example...
    """)
    return


@app.cell
def _():
    import numpy as np

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Load digits dataset
    X, y = load_digits(return_X_y=True)
    return MinMaxScaler, X, np, train_test_split, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have the data, now we need to find a way to encode it. Some dimensionality reduction maybe?
    """)
    return


@app.cell
def _(X):
    from sklearn.decomposition import PCA

    n_components = 2

    X_dim = PCA(n_components=n_components).fit_transform(X)
    return X_dim, n_components


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to make it easier we will focus only in two classes: $0$ and $8$.
    """)
    return


@app.cell
def _(X_dim, np, y):
    X_0 = X_dim[y == 0,:]
    X_8 = X_dim[y == 8,:]

    X_samples = np.vstack([X_0, X_8])
    y_samples = [0]*X_0.shape[0] + [8]*X_8.shape[0]
    return X_samples, y_samples


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we create our training and testing sets to provide some realistic metrics.
    """)
    return


@app.cell
def _(MinMaxScaler, X_samples, train_test_split, y_samples):
    # Split train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_samples,
        y_samples,
        test_size=0.33,
        random_state=42,
    )

    # Scale
    scaler = MinMaxScaler((-0.9, 0.9))
    X_train_1 = scaler.fit_transform(X_train)
    X_test_1 = scaler.transform(X_test)
    return X_test_1, X_train_1, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We proceed selecting a way in which our data will be encoded...
    """)
    return


@app.cell
def _(X_train_1):
    from squlearn.encoding_circuit import ChebyshevPQC

    # Encoding circuit
    encoding_circuit = ChebyshevPQC(num_qubits=X_train_1.shape[1], num_layers=1)
    encoding_circuit.draw(output="mpl", num_features=2)
    return (encoding_circuit,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Depending on the encoding we select, the size of the challenge may be quite difficult to handle on our devices. Lucky for us the people at Xanadu, IBM and other labs have spent some time implementing variants of classical simulators implementing GPU and TN execution backends that should help ease the task at hand.

    * Pennylane's Lighning plugins [Lightning plugins](https://docs.pennylane.ai/projects/lightning/en/stable/)
    * Qiskit Aer with its GPU and MPS simulators

    And sQUlearn supports most of them through their [Executor](https://squlearn.github.io/user_guide/executor.html) class.

    ![backends](https://squlearn.github.io/_images/executor.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So we could simply define the architecture of our Quantum Neural Network (a NN inspired version of a variational circuit in essence)

    ![qnn](https://squlearn.github.io/_images/qnn.svg)

    and focusing on our encoding circuit, free form ansatz and observables try to find the parameters that are able to learn the target class distribution simply focusing on the simulators that works best in our setup.
    """)
    return


@app.cell
def _(X_train_1, encoding_circuit, n_components, np, y_train):
    from squlearn import Executor
    from squlearn.optimizers import SLSQP
    from squlearn.observables import SinglePauli
    from squlearn.qnn import QNNClassifier, CrossEntropyLoss

    # Optimizer (gradient-free)
    slsqp = SLSQP(options={"maxiter": 100})

    # Define class
    clf = QNNClassifier(
        encoding_circuit,
        SinglePauli(n_components, 0),
        Executor(),
        CrossEntropyLoss(), # Loss
        slsqp, # Optimizer
        np.random.rand(16),
        np.random.rand(5)
    )

    # Train
    clf.fit(X_train_1, y_train)
    return CrossEntropyLoss, Executor, QNNClassifier, SinglePauli, clf


@app.cell
def _(X_test_1, clf, y_test):
    from sklearn.metrics import accuracy_score

    # Evaluate
    y_pred = clf.predict(X_test_1)
    print(f'Train accuracy score {accuracy_score(y_test, y_pred)}')
    return (accuracy_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We could check if GPU's might be of help when training a circuit that requires gradient computation...
    """)
    return


@app.cell
def _(
    CrossEntropyLoss,
    Executor,
    QNNClassifier,
    SinglePauli,
    X_train_1,
    encoding_circuit,
    n_components,
    np,
    y_train,
):
    import pennylane as qml
    from squlearn.optimizers import Adam
    from squlearn.qnn.util import get_lr_decay

    # Initialize the Executor with the PennyLane device
    dev = qml.device("lightning.gpu", wires=n_components)
    gpu_executor = Executor(dev, shots = 1234)

    # Optimizer (gradients)
    adam = Adam({'lr':get_lr_decay(0.01, 0.001, 100), 'maxiter': 10})

    # Define class
    clf_gpu = QNNClassifier(
        encoding_circuit,
        SinglePauli(n_components, 0),
        gpu_executor,
        CrossEntropyLoss(), # Loss
        adam, # Optimizer
        np.random.rand(16),
        np.random.rand(5)
    )

    # Train
    clf_gpu.fit(X_train_1, y_train)
    return clf_gpu, qml


@app.cell
def _(X_test_1, accuracy_score, clf_gpu, y_test):
    # Evaluate
    _y_pred = clf_gpu.predict(X_test_1)
    print(f'Train accuracy score {accuracy_score(y_test, _y_pred)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    So, what if we use an MPS simulator instead of a conventional state vector simulator? Our current stack would support this given that:

    * sQUlearn interfaces with Pennylane
    * Pennylane implements _default.tensor_ device on top of quimb
    * Quimb provides proficient TN simulation

    Although we might need for those three to align as they do not currently support it. Let's see how would go do if we wanted to build it ourselves... We know we can compose a circuit that runs using TNs for simulation tasks.
    """)
    return


@app.cell
def _(qml):
    # Initialize the device
    tensor_dev = qml.device("default.tensor", wires=2)

    @qml.qnode(tensor_dev)
    def circuit(weights):
        qml.RX(weights[0], wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(0))

    # Execute the circuit
    result = circuit([0.5, 0.5])
    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can take some inspiration from their tutorials on Hyrbid Neural Networks (classical and quantum layers): https://pennylane.ai/demos/tutorial_qnn_module_torch
    """)
    return


@app.cell
def _(np, qml):
    from pennylane import numpy as pnp

    class QuantumNeuralNetwork:
        """
        An enhanced Quantum Neural Network using PennyLane with fixes for learning issues.
        """

        def __init__(self, n_qubits=4, n_layers=2, device:str = "default.tensor"):
            """
            Initialize the QNN.

            Args:
                n_qubits: Number of qubits in the circuit
                n_layers: Number of layers in the ansatz
                device: PennyLane device to use
            """
            self.n_qubits = n_qubits
            self.n_layers = n_layers

            # Create device
            self.device = qml.device(device, wires=n_qubits)

            # Define the quantum circuit as a QNode with autograd interface
            @qml.qnode(self.device, interface="autograd")
            def circuit(weights, inputs=None):
                # Enhanced data encoding with proper scaling
                if inputs is not None:
                    for i in range(n_qubits):
                        qml.RY(inputs[i] * np.pi, wires=i)  # Scale to [0, π] range

                # Multiple variational layers for better expressivity
                for layer in range(n_layers):
                    # Parameterized rotations
                    for i in range(n_qubits):
                        qml.Rot(weights[layer, i, 0], weights[layer, i, 1], weights[layer, i, 2], wires=i)

                    # Enhanced entanglement pattern
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                    # Add circular entanglement for better connectivity
                    if n_qubits == 2:
                        qml.CNOT(wires=[1, 0])

                # Measurement - expectation value of Pauli Z on first qubit
                return qml.expval(qml.PauliZ(0))

            self.circuit = circuit

            # Better weight initialization with small values and requires_grad=True
            self.weights = pnp.array(np.random.normal(0, 0.1, (n_layers, n_qubits, 3)), requires_grad=True)

        def forward(self, inputs):
            """
            Forward pass of the QNN.

            Args:
                inputs: Classical input data

            Returns:
                Quantum circuit output (expectation values)
            """
            return self.circuit(self.weights, inputs=inputs)

        def train(self, X, y, n_epochs=100, learning_rate=0.1):
            """
            Train the QNN using enhanced gradient descent with learning fixes.

            Args:
                X: Input data (n_samples, n_features)
                y: Target values (n_samples,)
                n_epochs: Number of training epochs
                learning_rate: Initial learning rate
            """
            # Convert labels to -1, 1 for quantum classification
            y_quantum = np.where(np.array(y) == 0, -1, 1)

            # Hinge loss instead of square loss for improved training dynamics
            def hinge_loss(predictions, labels, margin=1.0):
                loss = 0
                for pred, label in zip(predictions, labels):
                    loss += max(0, margin - label * pred)
                return loss / len(labels)

            def loss(weights, X, y):
                predictions = np.array([self.circuit(weights, inputs=x) for x in X])
                return hinge_loss(predictions, y)

            # Training loop with enhancements
            for epoch in range(n_epochs):
                # Forward pass
                predictions = np.array([self.circuit(self.weights, inputs=x) for x in X])
                current_loss = hinge_loss(predictions, y_quantum)

                # Compute accuracy
                preds_sign = np.sign(predictions)
                current_acc = np.mean(preds_sign == y_quantum)

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Loss = {current_loss:.4f}, Accuracy = {current_acc:.3f}")

                # Backward pass
                grad = qml.grad(loss)(self.weights, X, y_quantum)
                grad_norm = pnp.linalg.norm(grad)

                # Gradient clipping to prevent exploding gradients
                if grad_norm > 1.0:
                    grad = grad / grad_norm

                # Learning rate scheduling
                current_lr = learning_rate * (0.9 ** (epoch // 10))

                # Update weights
                self.weights = self.weights - current_lr * grad

            # Final evaluation
            final_predictions = np.array([self.circuit(self.weights, inputs=x) for x in X])
            final_loss = hinge_loss(final_predictions, y_quantum)
            print(f"Training complete. Final loss = {final_loss:.4f}")

        def predict(self, inputs):
            """
            Make predictions using the trained QNN.

            Args:
                inputs: Input data to predict on

            Returns:
                Predictions from the QNN
            """
            return self.forward(inputs)

    return (QuantumNeuralNetwork,)


@app.cell
def _(
    QuantumNeuralNetwork,
    X_test_1,
    X_train_1,
    accuracy_score,
    n_components,
    y_test,
    y_train,
):
    # Use enhanced QNN with better parameters
    qnn = QuantumNeuralNetwork(n_qubits=n_components, n_layers=2, device="default.qubit")
    qnn.train(X_train_1, y_train, n_epochs=30, learning_rate=0.1)

    # Proper prediction thresholding for binary classification
    _y_pred = (qnn.predict(X_test_1.T) > 0.25)
    print(f'Test accuracy score {accuracy_score(y_test, _y_pred)}')
    return (qnn,)


@app.cell
def _(X_test_1, qml, qnn):
    qml.draw_mpl(qnn.circuit)(qnn.weights, inputs=X_test_1[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By a simple change of device we can try wiht a tensor network simulation...
    """)
    return


@app.cell
def _(
    QuantumNeuralNetwork,
    X_test_1,
    X_train_1,
    accuracy_score,
    n_components,
    y_test,
    y_train,
):
    # Use enhanced QNN with better parameters
    qnn_tn = QuantumNeuralNetwork(n_qubits=n_components, n_layers=1, device="default.tensor")
    qnn_tn.train(X_train_1, y_train, n_epochs=25, learning_rate=0.01)

    # Proper prediction thresholding for binary classification
    _y_pred = (qnn_tn.predict(X_test_1.T) > 0.25)
    print(f'Test accuracy score {accuracy_score(y_test, _y_pred)}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Such variation between training and test accuracy is one of the common issues with ML models, overfitting that also adds to some of the second most coommon issues related to QNN models, vanishing gradients and barren plateaus.

    We will need to improve our selection in order to prevent those issues but there is no nice path other than trying and reverse engineering why some of the things we changed workded.

    Also, simulator of choice will depend on how GPU intensive or paralellized the process might be so we will need to get familiarized with all combinations of:

    1. Model of choice
    2. Parameterization
    3. Best simulation/execution engine

    Happy coding!
    """)
    return


if __name__ == "__main__":
    app.run()
