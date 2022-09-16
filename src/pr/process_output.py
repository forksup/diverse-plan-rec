# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as plticker
from numpy import ndarray
import numpy as np
from matplotlib import pyplot


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
       for dirpath, _, filenames \
           in os.walk('/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/output_data/markov_data_output'):
        if len(filenames) == 0:
            continue
        data = defaultdict(dict)
        slices = {}
        overlapping = 0.150
        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1, len(filenames))]

        pyplot.rcParams["figure.figsize"] = [7.50, 3.50]
        pyplot.rcParams["figure.autolayout"] = True
        data = defaultdict(lambda: defaultdict(list))
        for j, f in enumerate(filenames):
            key = f.split("data")[2][1:].strip()
            ops = f.split("ops")[0][-1]
            blocks = f.split("blocks")[0][-1]
            gap = f.split("gap")[1][0]
            df = pd.read_csv(os.path.abspath(os.path.join(dirpath, f)))
            #df['C']>0)a
            len(df[df['top_k'] > 0])
            x = []
            y = []

            for i in range(20):
                size = len(df[df['top_k'] < i])
                if size:
                    y.append(len(df[df['top_k'] <= i])/len(df))
                    x.append(i)
            pyplot.xlabel("Prediction Count")
            pyplot.ylabel("Accuracy")

            pyplot.plot(x, y, label=f"{blocks}blocks & {gap} gap", lw=5, alpha=0.5, color=colors[j])

            data[blocks][gap] = [x,y]
        pyplot.legend()

        pyplot.show()
        exit(0)
        for blocks, d in data.items():
            for gap, (x,y) in d.items():
                pyplot.plot(x,y, label=f"{blocks}blocks & {gap} gap",lw=5, alpha=0.5)
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
            #plots[i].xaxis.set_major_locator(loc)
            i += 1
        plt.tight_layout()
        fig.suptitle(dirpath.split("/")[-1])  # or plt.suptitle('Main title')
        plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
