from matplotlib import pyplot as plt


def plot_roc_curve(fpr, tpr, area, label):
    lw = 2
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {area:2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {label} SET')
    plt.legend(loc="lower right")
    plt.show()
