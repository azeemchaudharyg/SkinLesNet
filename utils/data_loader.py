import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from config import IMAGE_SIZE

def load_data(directory, categories, image_size=IMAGE_SIZE):
    data = []
    
    for category in categories:
        path = os.path.join(directory, category)
        label = categories.index(category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                img_array = cv2.imread(img_path)
                img_array = cv2.resize(img_array, (image_size, image_size))
                data.append([img_array, label])
            except Exception as e:
                pass
    X, y = zip(*data)
    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
