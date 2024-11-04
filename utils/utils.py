import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

# Graph functions


class Utils:
    # Function to plot loss curve

    def plot_loss(self, train_loss, val_loss):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train loss')
        plt.plot(val_loss, label='Evaluate loss')
        plt.title('Learning curve during epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Funtion to plot accuracy

    def plot_accuracy(self, train_acc, val_acc):
        plt.figure(figsize=(10, 5))
        plt.plot(train_acc, label='Train accuracy')
        plt.plot(val_acc, label='Evaluate accuracy')
        plt.title('Accuracy curve during epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    # Function to plot class distribution

    def plot_class_distribution(self, labels):
        plt.figure(figsize=(8, 5))
        sns.countplot(x=labels)
        plt.title('Class distribution')
        plt.xlabel('Class')
        plt.ylabel('Frequency')
        plt.show()

    # Function to plot a histogram with the errors

    def plot_error_histogram(self, preds, labels):
        errors = preds.flatten() - labels.flatten()
        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=20, color='red', alpha=0.7)
        plt.title('Histogram (errors)')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
        plt.show()

    # Function to plot confusion matrix

    def plot_confusion_matrix(self, all_preds, all_labels):
        conf_matrix = sklearn_confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Classe 0', 'Classe 1'],
                    yticklabels=['Classe 0', 'Classe 1'])
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix')
        plt.show()

    # Function to plot probability distribution

    def plot_probability_distribution(self, preds):
        plt.figure(figsize=(10, 5))
        sns.histplot(preds, bins=30, kde=True)
        plt.title('Predicted probability distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.axvline(0.5, color='red', linestyle='dashed',
                    linewidth=2)
        plt.show()
