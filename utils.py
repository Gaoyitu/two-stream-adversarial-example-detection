import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np


def show_gray_image(image):
    plt.imshow(image,cmap="gray")
    plt.axis("off")
    plt.show()

def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()

def compute_roc(probs_neg,probs_pos):
    probs = np.concatenate((probs_neg,probs_pos))

    labels = np.concatenate((np.zeros_like(probs_neg),np.ones_like(probs_pos)))

    fpr,tpr,_ = roc_curve(labels,probs)
    auc_score = auc(fpr,tpr)

    return fpr,tpr,auc_score
