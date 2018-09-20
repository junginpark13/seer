import fnmatch
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np


def navigate_all_files(root_path, patterns):
    """
    A generator function that iterates all files that matches the given patterns
    from the root_path.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    elif isinstance(patterns, list):
        pass
    else:
        raise TypeError('Patterns must be either list or str instance.')
        
    for root, dirs, files in os.walk(root_path):
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                yield os.path.join(root, filename)

                
def get_all_files(root_path, patterns):
    """
    Returns a list of all files that matches the given patterns from the
    root_path.
    """
    ret = []
    for filepath in navigate_all_files(root_path, patterns):
        ret.append(filepath)
    return ret


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show_colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if show_colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
