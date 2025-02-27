from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from MultilayerPerceptron.mlp import *
import numpy as np
import pandas as pd

# fetch dataset
auto_mpg = fetch_ucirepo(id=9)

# data (as pandas dataframes)
X = auto_mpg.data.features
y = auto_mpg.data.targets

# Combine features and target into one DataFrame for easy filtering, then one-hot encode
data = pd.concat([X, y], axis=1)
data = pd.get_dummies(data, columns=['cylinders', 'model_year', 'origin'])

# Drop rows where the target variable is NaN
cleaned_data = data.dropna()

# Split the data back into features (X) and target (y)
y = cleaned_data["mpg"]
X = cleaned_data.drop(columns=["mpg"])

# Display the number of rows removed
rows_removed = len(data) - len(cleaned_data)
print(f"Rows removed: {rows_removed}")

# Do a 70/30 split (e.g., 70% train, 30% other)
X_train, X_leftover, y_train, y_leftover = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

# Split the remaining 30% into validation/testing (15%/15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_leftover, y_leftover,
    test_size=0.5,
    random_state=42,
    shuffle=True,
)

# Compute statistics for X (features) - mean and standard deviation
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)

# Standardize X
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

# Compute statistics for y (targets) - mean and standard deviation
y_mean = y_train.mean()
y_std = y_train.std()

# Standardize y
y_train = (y_train - y_mean) / y_std
y_val = (y_val - y_mean) / y_std
y_test = (y_test - y_mean) / y_std

# Define mlp for mpg dataset
mlp = MultilayerPerceptron((
        Layer(25, 15, Linear()),
        Layer(15, 15, Relu()),
        Layer(15, 1, Linear()),
    ))

# Convert to expected array shapes
y_train = np.array(y_train).reshape(-1, 1)
y_val = np.array(y_val).reshape(-1, 1)

training_error, val_error = mlp.train(np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), SquaredError(), rmsprop=True, epochs=32)
print("Final training error: ", training_error[-1])
print("Final validation error: ", val_error[-1])

# Pass test set to mlp for inference
y_pred = mlp.forward(np.array(X_test))

# Reshape y_test to match y_pred shape and calculate loss
y_test = np.array(y_test).reshape(-1, 1)
test_set_loss = np.mean(SquaredError().loss(y_test, y_pred))
print(f"Test set loss: {test_set_loss}")

# Randomly select 10 samples from test and prediction sets and undo standardization
random_sample_indices = np.random.choice(len(y_test), size=10, replace=False)
y_test_samples = (y_test[random_sample_indices] * y_std) + y_mean
y_pred_samples = (y_pred[random_sample_indices] * y_std) + y_mean

# Output 10 samples with true mpg vs. predicted mpg
for i in range(10):
    print(f"Test mpg: {y_test_samples[i][0]:.2f}, Predicted mpg: {y_pred_samples[i][0]:.2f}")