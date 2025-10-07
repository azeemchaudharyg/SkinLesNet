import numpy as np
import random
import matplotlib.pyplot as plt

def model_test():
    
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