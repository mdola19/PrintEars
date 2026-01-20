import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Loading & Processing the Datset ------------------------------------------------------------------
dataset_path = "MFCC_Features/dataset.npz"
data = np.load(dataset_path, allow_pickle=True)

X = data['X'] # shape = (num_clips (90), timesteps/clip (100), # of MFCC features (13))
y = data['y'] # shape = (num_clips (90),)
class_names = data['class_names'] # shape = (num_classes (5),)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", class_names)

# Adding another channel dimension to X
X = X[..., np.newaxis]  # New shape: (90, 100, 13, 1) # New axis represents coefficient value at each cell
print("X shape with channel dimension:", X.shape)

# Splitting Dataset: Training (70%), Validation (15%), Test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y) # random state makes split deterministic
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)
# --------------------------------------------------------------------------------------------------

wandb.init(project="PrintEar", config={
    "epochs": 50,
    "batch_size": 16,
    "architecture": "CNN",
    "dataset": "MFCC Features",
    "learning_rate": 0.001,
    "n_mfcc": 13,
})

# Building the Model -------------------------------------------------------------------------------
num_classes = len(class_names)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),  # Input Layer with shape: (100, 13, 1)

    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding="same"), # First Convolutional Layer with 16 kernels of size 3x3
    tf.keras.layers.MaxPooling2D((2, 2)), # First Max Pooling Layer using a 2x2 pool size

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same"), # Second Convolutional Layer with 32 kernels of size 3x3
    tf.keras.layers.MaxPooling2D((2, 2)), # Second Max Pooling Layer using a 2x2 pool size

    tf.keras.layers.Flatten(), # Flatten Layer to convert 2D feature maps to 1D feature vectors
    tf.keras.layers.Dense(64, activation='relu'), # Fully Connected Layer with 64 neurons
    tf.keras.layers.Dropout(0.5), # Dropout Layer to prevent overfitting

    tf.keras.layers.Dense(num_classes, activation='softmax') # Output Layer with softmax activation for multi-class classification
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
# --------------------------------------------------------------------------------------------------

# Training the Model -------------------------------------------------------------------------------
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), WandbMetricsLogger(log_freq="epoch")] # Enable early stopping

history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=16, 
                    validation_data=(X_val, y_val), 
                    callbacks=callbacks) # Start training

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

model.save("mfcc_cnn.keras")
print("Model Saved")
# --------------------------------------------------------------------------------------------------