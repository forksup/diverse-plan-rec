import math
import pickle
import string
from collections import defaultdict
import numpy
import pandas as pd
import os
from random import shuffle
from sklearn.model_selection import train_test_split
from string import punctuation

output_folder = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/test_domain_fruit_training_data"
frames = defaultdict(list)


for key in frames:
    shuffle(frames[key])


def load_data(df, testing=False, gap=6):
    colsToDrop = []
    for col in df.keys():
        if len(df[col].unique()) < 1 or "holding" in col or "arm" in col:
            colsToDrop.append(col)
    df = df.drop(columns=colsToDrop)

    # df = df.loc[[key for key, row in (df.shift() == df).iterrows() if not all(row)]]


    if not testing:
        df.loc[-1] = ["null" for _ in df.keys()]

    df = df.groupby(list(df.keys())).head(float('inf'))
    df.reset_index(drop=True, inplace=True)

    if testing:
        return rename_columns(df)

    s1, s2 = [-1], [0]

    for index, row in df.iterrows():
        if index == (len(df) - gap):
            break
        #for i in range(min(N,len(df)-index-2)):
        s1.append(index)
        s2.append(index + gap)

    # This feature is added to ensure domain of time slices are equal
    # However with 5k episodes we should be able to ensure that all outomes are included
    # We will have to find out

    # make sure not to add the null state more than once so we subtract one from the length
    #s1 += [len(df)- 2]
    #s2 += [-1 ]

    dataframeleft = df.iloc[s1]
    dataframeright = df.iloc[s2]

    dataframeright.append(pd.Series(), ignore_index=True)

    # now rename columns
    renameColumns = {}
    for col in df.keys():
        renameColumns[col] = str(col) + "0"
    dataframeleft = dataframeleft.rename(columns=renameColumns)
    dataframeleft.reset_index(drop=True, inplace=True)

    renameColumns = {}
    for col in df.keys():
        renameColumns[col] = str(col) + "1"
    dataframeright = dataframeright.rename(columns=renameColumns)
    dataframeright.reset_index(drop=True, inplace=True)

    # The problem is, is that when you concat multiple datafrmes there are columns which do not exist in the other.
    # So we fill those with various null values
    combined = pd.concat([dataframeleft, dataframeright], axis=1)

    combined = combined.replace(r'^\s*$', "null", regex=True)

    assert (len(combined) == len(dataframeleft))

    # Make sure domain of each sides aer the same
    # for key in dataframeleft.keys():
    # assert(len(dataframeleft[key].unique()) == len(dataframeright[key[:-1]+"1"].unique()))

    return rename_columns(combined)

def rename_columns(combined):
    keysToRename = {}
    for k, r in combined.iterrows():
        for col in combined.keys():
            if "-" in col:
                keysToRename[col] = col.replace("-", "X")
            if isinstance(r[col], str):
                r[col] = r[col].strip(punctuation)
                r[col] = r[col].replace("-", "X")

    return combined.rename(keysToRename, axis=1)  #

def process_solutions(frame, f_key):



    preprocessed_training = frame.iloc[:math.ceil(len(frame)*.7)]
    test_dataset = frame.iloc[-math.ceil(len(frame)*.3):]

    input_file = open(r'data_split.obj', 'wb')
    #split = pickle.load(input_file)
    pickle.dump([preprocessed_training,test_dataset], input_file)


    test_dataset = load_data(test_dataset, True)
    test_dataset.to_csv(f"{output_folder}/test_{f_key}.txt", sep=",", index=False)

    for gap in range(1, 5):
        print("Data Gap:" + str(gap))

        solutions = []
        # Make sure there are more rows than gap, otherwise it will be incomplete
        training_dataset = load_data(preprocessed_training, False, gap)

        # Make sure domain of each sides aer the same
        for key in training_dataset.keys():
            if "0" in key:
                assert (len(training_dataset[key].unique()) == len(training_dataset[key[:-1] + "1"].unique()))

        training_dataset = fillNull(training_dataset)

        training_dataset.to_csv(f"{output_folder}/train_data_gap{gap}_{f_key}.txt", sep=",", index=False)

        for key in training_dataset.keys():
            #Subtract one for end in test dataset
            assert(len(test_dataset[key[:-1]].unique()) <= len(training_dataset[key].unique()))


def fillNull(combined):
    for key in combined.keys():
        combined[key] = combined[key].fillna('null')

        """
        if "Hands" in key:
            combined[key] = combined[key].fillna('Empty')
        elif "close" in key.lower():
            combined[key] = combined[key].fillna('null')
        elif "act" in key.lower():
            combined[key] = combined[key].fillna('none')
        elif "True" in combined[key].unique():
            combined[key] = combined[key].fillna('False')
        else:
            combined[key] = combined[key].fillna('null')
            """
    return combined

def runThroughFrames():
    df = pd.read_csv("test_output.txt")

    process_solutions(df, "test")

if __name__ == "__main__":
    runThroughFrames()
"""
path=/home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/benchmarks/depot/
domain=/mnt/watchandhelp/PycharmProjects/pddl-generators/blocksworld/3ops/domain.pddl
cd $path
for file in *;
 do
     if [[ $file == "task"* ]]; then
      python /home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/src/pyperplan.py "$domain" "$path/$file" -s astar
    fi;
done

      python /home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/src/pyperplan.py "/home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/benchmarks/freecell" "/home/mitch/PycharmProjects/diverse-plan-rec/datasets/campus/bui-campus_generic_hyp-0_10_1/template.pddl"
      python /home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/src/pyperplan.py "$path/domain.pddl" "$path/$file" -s ehs



path=/mnt/watchandhelp/PycharmProjects/pddl-generators/blocksworld/data/data-3blocks-4ops
domain=/mnt/watchandhelp/PycharmProjects/pddl-generators/blocksworld/4ops/domain.pddl
cd $path
for file in *;
 do
    python /home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/src/pyperplan.py "$domain" "$path/$file" -s astar
done
"""
