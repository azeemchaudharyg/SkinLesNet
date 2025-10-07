import numpy as np
import random
import matplotlib.pyplot as plt
from utils.data_loader import load_data
from utils.plots import plot_history, plot_confusion_matrix, data_distribution, sample_images
from models.skinlesnet_model import skinlesnet
from tensorflow.keras.optimizers import Adam

from config import BATCH_SIZE, EPOCHS, IMAGE_SIZE, DIRECTORY, CATEGORIES, LEARNING_RATE




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
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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


# Test the model
print("##############----Testing the Model----##############\n")

idx2 = random.randint(0, len(y_test))

test_img = X_test[idx2, :]
test_label = y_test[idx2]

y_pred = model.predict(X_test[idx2,:].reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))
y_pred = np.argmax(y_pred)

if (y_pred == 0):
    pred = 'Melanoma'
elif (y_pred == 1):
    pred = 'Nevus'
else:
    pred = 'Seborrheic Keratosis'
 
if (test_label == 0):
    plt.title("Actual Image: Melanoma" +"\nModel Prediction: " + str(pred))
elif (test_label == 1):
    plt.title("Actual Image: Nevus" +"\nModel Prediction: " + str(pred))
else:
    plt.title("Actual Image: Seborrheic Keratosis" +"\nModel Prediction: " + str(pred))
        
plt.imshow(test_img)
plt.show
