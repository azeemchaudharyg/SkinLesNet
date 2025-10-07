import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def plot_history(history):
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.legend()
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, target_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, cmap="flare", fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_true, y_pred, target_names=target_names))
    

def data_distribution(lesions_data):

    plot_list = ["Melanoma" if i[1]==0 else "Nevus" if i[1]==1 else "Seborrheic Keratosis" for i in lesions_data]

    ax = sns.countplot(plot_list)
    plt.title("Total Images in Each Class")
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}', (p.get_x() + 0.25, p.get_height() + 0.01))

    counts = {cat: plot_list.count(cat) for cat in set(plot_list)}
    categories = counts.keys()
    category_counts = counts.values()

    category_counts_array = np.array(list(category_counts))
    mean_value = np.mean(category_counts_array)
    median_value = np.median(category_counts_array)
    std_deviation = np.std(category_counts_array)

    statistical_info = f'Mean: {mean_value:.2f}\nMedian: {median_value:.2f}\nStd Dev: {std_deviation:.2f}'
    clinical_implications = (
        "The distribution of skin lesion categories\n"
        "may indicate varying prevalence rates,\n"
        "potentially influencing diagnostic priorities.\n"
        "Notably, the higher standard deviation\n"
        "suggests significant variability in lesion\n"
        "presentation, warranting thorough examination."
    )

    plt.figure(figsize=(6, 6))
    plt.pie(category_counts, labels=categories, autopct='%1.1f%%')
    plt.text(1.2, 0.5, statistical_info, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center')
    plt.text(1.2, 0.2, clinical_implications, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center')
    plt.axis('equal')
    plt.title('Skin Lesions Distribution with Clinical Annotations')
    plt.show()
    
    
    def sample_images(X, y, class_names, num_images=15):
        rows, cols = 5, 5
        fig = plt.figure(figsize=(12, 12))
        for i in range(1, num_images + 1):
            fig.add_subplot(rows, cols, i)
            plt.imshow(X[i - 1])
            plt.xticks([])
            plt.yticks([])
            plt.title(class_names[y[i - 1]])
        plt.show()


