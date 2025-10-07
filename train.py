import numpy as np
from utils.data_loader import load_data
from utils.plots import plot_history, plot_confusion_matrix, data_distribution, sample_images
from models.skinlesnet_model import skinlesnet
from tensorflow.keras.optimizers import Adam


# Configuration
DIRECTORY = "data/"    # Put dataset in the data folder and path here.....!!!!
CATEGORIES = ['melanoma', 'nevus', 'seborrheic_keratosis']   # Sub directories in the main data folder
BATCH_SIZE = 32
EPOCHS = 100



# Load data
X_train, X_test, y_train, y_test = load_data(DIRECTORY, CATEGORIES)
print("Training samples:", X_train.shape, "Testing samples:", X_test.shape)

# Combine train and test sets into one list for visualization
lesions_data = list(zip(np.concatenate([X_train, X_test]),
                        np.concatenate([y_train, y_test])))

data_distribution(lesions_data)

# Display first 15 images
sample_images(X_train, y_train, CATEGORIES)

# Build and compile model
model = skinlesnet()
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {acc*100:.2f}%")

# Plots
plot_history(history)
y_pred = np.argmax(model.predict(X_test), axis=1)
plot_confusion_matrix(y_test, y_pred, CATEGORIES)

# Save model
model.save("models/skinlesnet_model.h5")
