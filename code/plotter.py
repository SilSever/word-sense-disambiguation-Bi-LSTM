import os
from typing import List, Dict

import matplotlib.pyplot as plt
from tensorflow.python.keras import Model

import config

"""
    :author Silvio Severino
"""


def plot_train(history: Model) -> None:
    """
    A simple method to plot the accuracy and loss result
    :param history: history of training
    """
    train_metrics = {
        0: ["dense_bn_loss", "val_dense_bn_loss", "dense_bn_acc", "val_dense_bn_acc"],
        1: [
            "dense_wnd_loss",
            "val_dense_wnd_loss",
            "dense_wnd_acc",
            "val_dense_wnd_acc",
        ],
        2: [
            "dense_lex_loss",
            "val_dense_lex_loss",
            "dense_lex_acc",
            "val_dense_lex_acc",
        ],
    }

    if not os.path.exists(config.PLOT_FOLDER):
        os.mkdir(config.PLOT_FOLDER)

    for i in train_metrics:
        _plot_tr(
            history,
            loss=train_metrics[i][0],
            val_loss=train_metrics[i][1],
            acc=train_metrics[i][2],
            val_acc=train_metrics[i][3],
        )


def _plot_tr(history: Model, loss: str, val_loss: str, acc: str, val_acc: str) -> None:
    """
    A simple method to plot the accuracy and loss result
    :param history: history of training
    :param loss: loss name
    :param val_loss: validation loss name
    :param acc: accuracy name
    :param val_acc: validation accuracy name
    :return:
    """
    domain = loss.split("_")[-2]
    fig, axes = plt.subplots(2, sharex="all", figsize=(12, 8))
    axes[0].set_ylabel(domain.upper() + " Loss", fontsize=14)
    axes[0].plot(history.history[loss])
    axes[0].plot(history.history[val_loss])
    axes[0].legend(
        [loss, val_loss],
        loc="upper right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    axes[1].set_ylabel(domain.upper() + " Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(history.history[acc])
    axes[1].plot(history.history[val_acc])
    axes[1].legend(
        [acc, val_acc],
        loc="lower right",
        frameon=True,
        facecolor="white",
        fontsize="large",
    )

    plt.savefig(os.path.join(config.PLOT_FOLDER, domain + ".png"))
    plt.show()


def plot_frequencies(frequencies) -> None:
    """
    A simple method to handle the plotting of the frequency distribution
    :param frequencies: frequency distribution
    :return:
    """
    for freq, title in frequencies:
        _plot(freq, 45, title=title)


def _plot(self, *args, **kwargs) -> None:
    """
    Plot samples from the frequency distribution
    displaying the most frequent sample first.  If an integer
    parameter is supplied, stop after this many samples have been
    plotted.  For a cumulative plot, specify cumulative=True.
    (Requires Matplotlib to be installed.)

    :param title: The title for the graph
    :type title: str
    :param cumulative: A flag to specify whether the plot is cumulative (default = False)
    :type title: bool
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ValueError(
            "The plot function requires matplotlib to be installed."
            "See http://matplotlib.org/"
        )

    if len(args) == 0:
        args = [len(self)]
    samples = [item for item, _ in self.most_common(*args)]

    cumulative = _get_kwarg(kwargs, "cumulative", False)
    percents = _get_kwarg(kwargs, "percents", False)
    if cumulative:
        freqs = list(self._cumulative_frequencies(samples))
        ylabel = "Cumulative Counts"
        if percents:
            freqs = [f / freqs[len(freqs) - 1] * 100 for f in freqs]
            ylabel = "Cumulative Percents"
    else:
        freqs = [self[sample] for sample in samples]
        ylabel = "Counts"
    # percents = [f * 100 for f in freqs]  only in ProbDist?

    ax = plt.gca()
    ax.grid(True, color="silver")

    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 2
    if "title" in kwargs:
        title = kwargs["title"]
        ax.set_title(title)
        del kwargs["title"]

    ax.plot(freqs, **kwargs)
    ax.set_xticks(range(len(samples)))
    ax.set_xticklabels([str(s) for s in samples], rotation=90)
    ax.set_xlabel("Samples")
    ax.set_ylabel(ylabel)

    plt.savefig(
        os.path.join(str(config.PLOT_FOLDER), "freq_" + title + ".png"),
        bbox_inches="tight",
    )
    plt.show()


def _get_kwarg(kwargs, key, default):
    """
    A method to handle _plot parameters
    """
    if key in kwargs:
        arg = kwargs[key]
        del kwargs[key]
    else:
        arg = default
    return arg


def plot_corpus_train(
    train_x: List,
    dev_x: List,
    tr_sens: List,
    pos: List,
    vocab: Dict[str, int],
    lab_bn: List,
    dev_lab_bn: List,
    sens_inv_bn: Dict[str, int],
    lab_wnd: List,
    dev_lab_wnd: List,
    sens_inv_wnd: Dict[str, int],
    lab_lex: List,
    dev_lab_lex: List,
    sens_inv_lex: Dict[str, int],
    lab_pos: List,
    dev_lab_pos: List,
    sens_inv_pos: Dict[str, int],
) -> None:
    """
    A simple method to plot the summary of training sentences
    :param train_x: training input
    :param dev_x: development input
    :param tr_sens: sense_embeddings phrase input
    :param pos: pos input
    :param lab_bn: babelnet labels
    :param dev_lab_bn: development babelnet labels
    :param lab_wnd: wordnet domains labels
    :param dev_lab_wnd: development wordnet domains labels
    :param lab_lex: lexicographer labels
    :param dev_lab_lex: development lexicographer labels
    :param lab_pos: pos labels
    :param dev_lab_pos: development pos labels
    :param vocab: training vocab
    :param sens_inv_bn: babelnet sense inventory
    :param sens_inv_wnd: wordnet domains sense inventory
    :param sens_inv_lex: lexicographer sense inventory
    :param sens_inv_pos: pos sense inventory
    :return: None
    """
    print("KIND\t\t\t\t", "tr\t\t", "dev\t" "voc")
    print("INPUT\t\t\t\t", len(train_x), "\t", len(dev_x), "\t", len(vocab))
    print(
        "BABELNET DOMAIN\t\t",
        len(lab_bn),
        "\t",
        len(dev_lab_bn),
        "\t",
        len(sens_inv_bn),
    )
    print(
        "WORDNET DOMAIN\t\t",
        len(lab_wnd),
        "\t",
        len(dev_lab_wnd),
        "\t",
        len(sens_inv_wnd),
    )
    print(
        "LEX DOMAIN\t\t\t",
        len(lab_lex),
        "\t",
        len(dev_lab_lex),
        "\t",
        len(sens_inv_lex),
    )
    print("POS\t\t\t", len(lab_pos), "\t", len(dev_lab_pos), "\t", len(sens_inv_pos))

    print("\nExample...")
    print("\ttrain_x: ", train_x[2])
    print("\ttr_sens: ", tr_sens[2])
    print("\tpos_x: ", pos[2])
    print("\tlab_bn: ", lab_bn[2])
    print("\tlab_wnd: ", lab_wnd[2])
    print("\tlab_lex: ", lab_lex[2])
    print("\tpos_y: ", lab_pos[2])
