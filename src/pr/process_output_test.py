# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict
import os
import pandas as pd
import matplotlib.ticker as plticker
from numpy import ndarray
import numpy as np


import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.025
fig = plt.subplots(figsize=(15, 8))

def convert_row(row):
    return [row[i] for i in range(0, len(row), 2)]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = defaultdict(lambda: defaultdict(list))
    plt.title("fruit domain results")
    for dirpath, _, filenames \
            in os.walk('/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/output_data/markov_data_output'):
        if len(filenames) == 0:
            continue
        slices = {}
        overlapping = 0.150
        colormap = plt.cm.nipy_spectral

        plt.rcParams["figure.figsize"] = [10.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        for j, f in enumerate(filenames):
            df = pd.read_csv(os.path.abspath(os.path.join(dirpath, f)), index_col=False)
            data[f] = df
    res = []
    keys = []
    x = []
    for key in df.keys():
        results = []
        if "top" in key and len(df[key].dropna()) > 0:
            for filename in sorted(data.keys()):
                x.append(filename)
                df = data[filename]
                results.append(len(df[df[key] == 0]) / len(df))
            keys.append(key[:-6])
            res.append(results)

    #keys.append("state_prediction")
    acc = []
    for filename in sorted(data.keys()):
        results = []
        df = data[filename]
        for _, row in df.iterrows():
            # If all topk are 0 this returns true
            results.append(not any(convert_row(row)))
        # Now let's count when all topk are 0
        acc.append(results.count(True) / len(results))
    print(acc)
    res.append(acc)
    keys.append("state_prediction")

    br = [np.arange(len(res[0]))]
    for i in range(1, len(res)):
        br.append([x + barWidth for x in br[i - 1]])

    for i in range(len(res)):
        plt.bar(br[i], res[i], width=barWidth,
                edgecolor='grey', label=keys[i])
    # Adding Xticks
    plt.xlabel('Model Count', fontweight='bold', fontsize=15)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(res[0]))],
               sorted(data.keys()))

    plt.legend()
    plt.show()
    exit(0)

    for dirpath, _, filenames \
            in os.walk('/src/pr/fruit_domain_results'):
        if len(filenames) == 0:
            continue
        slices = {}
        overlapping = 0.150
        colormap = plt.cm.nipy_spectral

        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        for j, f in enumerate(filenames):
            key = f.split("data")[-1][1:].strip()
            # ops = f.split("ops")[0][-1]
            # blocks = f.split("blocks")[0][-1]
            gap = f.split("gap")[1][0]
            df = pd.read_csv(os.path.abspath(os.path.join(dirpath, f)), index_col=False)
            data[gap] = df

    for key in df.keys():
        x = [i + 1 for i in range(len(data))]
        y = [None for _ in range(len(data))]
        for gap, df in data.items():
            if "top" in key:
                size = len(df[df[key] == 0])

                if size:
                    y[int(gap) - 1] = size / len(df)
                else:
                    y[int(gap) - 1] = 0
        plt.xticks(np.arange(min(x), max(x) + 1, 1.0))
        plt.xlabel("Data Gap")
        plt.ylabel("Accuracy")
        plt.title("Output of Test Using Sampling")
        plt.plot(x, y, label=f"{key}")

    # data[blocks][gap] = [x,y]
    plt.legend()

    plt.show()

    exit(0)
    for blocks, d in data.items():
        for gap, (x, y) in d.items():
            pyplot.plot(x, y, label=f"{blocks}blocks & {gap} gap", lw=5, alpha=0.5)
        pyplot.legend()
        pyplot.show()

    exit(0)
    sizeOfDataset = len(data) // 2
    # create data

    fig, plots = plt.subplots(sizeOfDataset, 1)
    if not isinstance(plots, ndarray):
        plots = [plots]
    fig.suptitle(dirpath.split("/")[-1])
    i = 0;
    loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals

    for key, row in data.items():
        gap = list(row.keys())
        gap.sort()
        gap = [float(g) for g in gap]
        y = list(range(len(gap)))
        for key2, row2 in row.items():
            y[gap.index(float(key2))] = float(row2)
        plots[i // 2].xaxis.set_major_locator(loc)
        plots[i // 2].plot(gap, y, label=key)
        plots[i // 2].set_ylabel('accuracy')
        plots[i // 2].set_xlabel('data gap')
        plots[i // 2].legend()
        i += 1
    plt.tight_layout()
    plt.show()

    from matplotlib.pyplot import figure

    fig, plots = plt.subplots(len(slices), 1)
    fig.set_size_inches(9, 30, forward=True)
    i = 0;
    loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals

    for key, sl in slices.items():
        plots[i].title.set_text(key)
        plots[i].hist(sl, density=True, bins=30)  # density=False would make counts
        # ax.xaxis.set_major_locator(loc)

        plots[i].set_ylabel('Probability')
        plots[i].set_xlabel('Gap')
        # plots[i].xaxis.set_major_locator(loc)
        i += 1
    plt.tight_layout()
    fig.suptitle(dirpath.split("/")[-1])  # or plt.suptitle('Main title')
    plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
