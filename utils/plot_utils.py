from sklearn.metrics import roc_curve, RocCurveDisplay, precision_recall_curve,\
    PrecisionRecallDisplay
from matplotlib.axes import Axes
import numpy as np


def plot_roc(ax: Axes, pred: np.array, gth: np.array):
    fpr, tpr, _ = roc_curve(gth, pred)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot(ax=ax)


def plot_pr(ax: Axes, pred: np.array, gth: np.array):
    prec, recall, _ = precision_recall_curve(gth, pred)
    pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)
    pr_display.plot(ax=ax)
