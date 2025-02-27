import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


eps = 1e-8

def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.
    :param train_x: (np.ndarray) Input features of shape (n, f).
    :param train_y: (np.ndarray) Target values of shape (n, q).
    :param batch_size: (int) The size of each batch.
    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    for i in range(0, len(train_x), batch_size):
        yield train_x[i:i + batch_size], train_y[i:i + batch_size]

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x
        Input args may differ in the case of softmax
        :param x: (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x: (np.ndarray) input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        g_x = self.forward(x)
        return g_x * (1 - g_x)

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x) + eps)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        g_x = self.forward(x)
        return 1 - g_x ** 2

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray, axis=-1) -> np.ndarray:
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=axis, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)
        batch_size, num_classes = s.shape

        # Initialize Jacobian tensor
        jacobian = np.zeros((batch_size, num_classes, num_classes))

        # Convert each batch entry to column vector and compute its jacobian
        for i in range(batch_size):
            s_i = s[i].reshape(-1, 1)
            jacobian[i] = np.diag(s_i) - np.dot(s_i, s_i.T)

        return jacobian

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.mean(0.5 * (y_pred - y_true) ** 2, axis=0)

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -np.einsum('bi,bi->b', y_true, np.log(y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction, dropout_rate: float = 0):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate

        # this will store the pre- and post- activations (forward prop)
        self.z = None
        self.activations = None

        # this will store dropout mask
        self.mask = None

        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biases
        glorot_boundary = np.sqrt(6 / (fan_in + fan_out))
        self.W = np.random.uniform(low=-glorot_boundary, high=glorot_boundary, size=(fan_in, fan_out))  # weights
        self.b = np.zeros(fan_out)  # biases

    def forward(self, h: np.ndarray, is_training=False):
        """
        Computes the activations for this layer
        :param h: input to layer
        :param is_training: (bool) Whether the layer is training or not (defaults to false)
        :return: layer activations
        """
        self.z = h @ self.W + self.b
        self.activations = self.activation_function.forward(self.z)

        # Apply dropout mask to activations
        if is_training and self.dropout_rate > 0:
            mask = (np.random.rand(*self.activations.shape) > self.dropout_rate).astype(np.float32)
            self.mask = mask
            self.activations = self.mask * self.activations / (1.0 - self.dropout_rate)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients
        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        dO_dZ = self.activation_function.derivative(self.z)

        # Mask delta using dropout mask if present
        if self.dropout_rate > 0:
            delta *= self.mask

        # Utilize softmax derivation with cross entropy shortcut assuming softmax is only in final layer
        if isinstance(self.activation_function, Softmax):
            delta_had_dO_dZ = delta
        else:
            delta_had_dO_dZ = delta * dO_dZ

        dL_dW = h.T @ delta_had_dO_dZ
        dL_db = np.sum(delta_had_dO_dZ, axis=0)
        self.delta = delta_had_dO_dZ @ self.W.T
        return dL_dW, dL_db

class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer, ...]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray, is_training=False) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :param is_training: (bool) Whether the network is training or not (defaults to false)
        :return: network output
        """
        # Calculate activations from first layer
        activations = self.layers[0].forward(x, is_training=is_training)

        # Iterate through remaining layers to update activations using previous layer's activations
        for i in range(1, len(self.layers)):
            activations = self.layers[i].forward(activations, is_training=is_training)

        return activations

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []

        delta = loss_grad
        last_layer_index = len(self.layers) - 1

        # Handle backprop computation for single layer edge-case
        if len(self.layers) == 1:
            dL_dw, dL_db = self.layers[last_layer_index].backward(input_data, delta)
            dl_dw_all.insert(0, dL_dw)
            dl_db_all.insert(0, dL_db)
            return dl_dw_all, dl_db_all

        dL_dw, dL_db = self.layers[last_layer_index].backward(self.layers[last_layer_index - 1].activations, delta)
        delta = self.layers[last_layer_index].delta
        dl_dw_all.insert(0, dL_dw)
        dl_db_all.insert(0, dL_db)

        # Iterate through layers in reverse order
        for i in range(len(self.layers) - 2, -1, -1):
            if i > 0:
                dL_dw, dL_db = self.layers[i].backward(self.layers[i - 1].activations, delta)
                delta = self.layers[i].delta
                dl_dw_all.insert(0, dL_dw)
                dl_db_all.insert(0, dL_db)

            # Handle first network layer separately
            else:
                dL_dw, dL_db = self.layers[i].backward(input_data, delta)
                dl_dw_all.insert(0, dL_dw)
                dl_db_all.insert(0, dL_db)

        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
              loss_func: LossFunction, learning_rate: float = 1E-3, batch_size: int = 16, epochs: int = 32,
              rmsprop=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron
        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :param rmsprop: whether to use RMSProp for optimization
        :return: list of training losses and validation losses
        """
        training_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)

        # If using rmsprop, initialize ndarrays for holding moving avg of gradients squared
        if rmsprop:
            rmsprop_moving_avg_W = [np.zeros_like(layer.W) for layer in self.layers]
            rmsprop_moving_avg_b = [np.zeros_like(layer.b) for layer in self.layers]

        for epoch in range(epochs):
            epoch_training_losses = []
            epoch_val_losses = []

            for train_batch in batch_generator(train_x, train_y, batch_size):
                x_batch, y_batch = train_batch
                y_pred = self.forward(x_batch, is_training=True)

                y_batch = y_batch.reshape(y_batch.shape[0], -1)
                y_pred = y_pred.reshape(y_pred.shape[0], -1)

                training_loss = loss_func.loss(y_batch, y_pred)
                epoch_training_losses.append(np.mean(training_loss))
                loss_grad = loss_func.derivative(y_batch, y_pred)

                dl_dw_all, dl_db_all = self.backward(loss_grad, x_batch)

                for i in range(len(self.layers)):
                    if rmsprop:
                        alpha = 0.9
                        rmsprop_moving_avg_W[i] = alpha * rmsprop_moving_avg_W[i] + (1 - alpha) * dl_dw_all[i] ** 2
                        rmsprop_moving_avg_b[i] = alpha * rmsprop_moving_avg_b[i] + (1 - alpha) * dl_db_all[i] ** 2

                        self.layers[i].W -= (learning_rate / np.sqrt(rmsprop_moving_avg_W[i] + eps)) * dl_dw_all[i]
                        self.layers[i].b -= (learning_rate / np.sqrt(rmsprop_moving_avg_b[i] + eps)) * dl_db_all[i]
                    else:
                        self.layers[i].W -= learning_rate * dl_dw_all[i]
                        self.layers[i].b -= learning_rate * dl_db_all[i]

            training_losses[epoch] = np.mean(np.array(epoch_training_losses))

            for val_batch in batch_generator(val_x, val_y, batch_size):
                x_batch, y_batch = val_batch
                y_pred = self.forward(x_batch)
                y_batch = y_batch.reshape(y_batch.shape[0], -1)
                y_pred = y_pred.reshape(y_pred.shape[0], -1)
                val_loss = loss_func.loss(y_batch, y_pred)
                epoch_val_losses.append(np.mean(val_loss))

            validation_losses[epoch] = np.mean(np.array(epoch_val_losses))

            print(
                f"Epoch {epoch + 1}: Training Loss = {training_losses[epoch]:.4f}, Validation Loss = {validation_losses[epoch]:.4f}")

        # Create training & validation loss graph
        plt.figure(figsize=(8, 5))
        plt.plot(range(epochs), training_losses, label='Training Loss', linestyle='-', marker='o')
        plt.plot(range(epochs), validation_losses, label='Validation Loss', linestyle='--', marker='s')

        # Add labels and title
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs. Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.show()

        return training_losses, validation_losses
