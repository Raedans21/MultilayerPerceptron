import struct
from sklearn.model_selection import train_test_split
from MultilayerPerceptron.mlp import *

# MnistDataLoader code from https://www.kaggle.com/code/hojjatk/read-mnist-dataset
class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        """ Reads MNIST images and labels from .idx files. """

        # Load labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = np.frombuffer(file.read(), dtype=np.uint8)

        # Load images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = np.frombuffer(file.read(), dtype=np.uint8).reshape(size, rows * cols)


        return image_data, labels

    def load_data(self):
        """ Loads MNIST data and splits training into training and validation sets (80/20). """

        # Load the datasets
        x_train_set, y_train_set = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test_set, y_test_set = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)

        # Compute mean and standard deviation from the training set
        x_mean = np.mean(x_train_set, axis=0)
        x_std = np.std(x_train_set, axis=0)

        # Account for pixels with no variance, setting their std to 1 to defuse their impact and avoid overflow
        x_std = np.where(x_std == 0, 1, x_std)

        # Standardize datasets (subtract mean, divide by std)
        x_train_set = (x_train_set - x_mean) / x_std
        x_test_set = (x_test_set - x_mean) / x_std

        # Split training data into 80% training and 20% validation
        x_train_set, x_val_set, y_train_set, y_val_set = train_test_split(
            x_train_set, y_train_set, test_size=0.2, random_state=42, shuffle=True
        )

        # One hot encode labels
        y_train_set, y_val_set, y_test_set = [np.eye(10)[y] for y in (y_train_set, y_val_set, y_test_set)]

        return (x_train_set, y_train_set), (x_val_set, y_val_set), (x_test_set, y_test_set)

class ModelUtils:
    def __init__(self, x_test, y_test, y_pred):
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = y_pred

    def calculate_accuracy(self):
        """
        :return: Percentage of correctly classified images
        """

        y_pred_labels = np.argmax(self.y_pred, axis=1)

        # Convert one-hot encoded true labels into class labels
        y_true_labels = np.argmax(self.y_test, axis=1)

        total_wrong_count = 0
        for i in range(len(y_pred_labels)):
            if y_pred_labels[i] != y_true_labels[i]:
                total_wrong_count += 1

        print(f"Total: {len(y_true_labels)}, total wrong: {total_wrong_count}")
        # Compute accuracy: Percentage of correct predictions
        accuracy = np.mean(y_pred_labels == y_true_labels)

        return accuracy

    def get_class_sample_predictions(self):
        """
        Selects one image per class (0-9) and displays them with the predicted label.
        :param x_test: numpy array of test input images
        :param y_test: numpy array of test output labels
        :param y_pred: numpy array of predicted labels
        """

        # Get model predictions
        y_pred_labels = np.argmax(self.y_pred, axis=1)  # Convert softmax output to label predictions
        y_true_labels = np.argmax(self.y_test, axis=1)  # Convert one-hot encoded labels to class numbers

        # Dictionary to store one example per class
        selected_samples = {}

        for i in range(len(y_true_labels)):
            label = y_true_labels[i]  # True class label
            if label not in selected_samples:  # Only store one example per class
                selected_samples[label] = (self.x_test[i], y_pred_labels[i])
            if len(selected_samples) == 10:  # Stop once we have one example per digit
                break

        # Sort samples into true label order
        selected_samples = sorted(selected_samples.items())

        # Plot images
        fig, axes = plt.subplots(2, 5, figsize=(10, 5))
        fig.suptitle("Predicted Class for Each Digit (0-9)", fontsize=14)

        for i, (label, (image, pred)) in enumerate(selected_samples):
            ax = axes[i // 5, i % 5]
            ax.imshow(image.reshape(28, 28), cmap="gray")  # Reshape to 28x28
            ax.set_title(f"MLP Prediction: {pred}", fontsize=10)
            ax.axis("off")

        plt.show()


# File paths for MNIST dataset
training_images_filepath = "../train-images.idx3-ubyte"
training_labels_filepath = "../train-labels.idx1-ubyte"
test_images_filepath = "../t10k-images.idx3-ubyte"
test_labels_filepath = "../t10k-labels.idx1-ubyte"

# Load and split data
mnist_loader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                               test_labels_filepath)
(x_train, y_train), (x_val, y_val), (x_test, y_test) = mnist_loader.load_data()

# Print shapes to verify
print("Training set shape: ", x_train.shape, y_train.shape)
print("Validation set shape: ", x_val.shape, y_val.shape)
print("Test set shape: ", x_test.shape, y_test.shape)

mlp = MultilayerPerceptron((
    Layer(784, 256, Relu(), dropout_rate=0.2),
    Layer(256, 128, Sigmoid()),
    Layer(128, 64, Relu()),
    Layer(64, 10, Softmax())
))

training_error, val_error = mlp.train(np.array(x_train), np.array(y_train), np.array(x_val), np.array(y_val), CrossEntropy(), rmsprop=True, epochs=10)
print("Final training error: ", training_error[-1])
print("Final validation error: ", val_error[-1])

y_pred = mlp.forward(x_test)

model_utils = ModelUtils(x_test, y_test, y_pred)
mnist_accuracy = model_utils.calculate_accuracy()
print(f"Accuracy on test set: {mnist_accuracy*100:.2f}%")
model_utils.get_class_sample_predictions()
