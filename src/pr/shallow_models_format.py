import pickle
import string
from collections import defaultdict

import pandas as pd
import os
from random import shuffle
from sklearn.model_selection import train_test_split
from string import punctuation

output_folder = "output_underlying_plans"
directory = "/mnt/watchandhelp/PycharmProjects/pddl-generators/blocksworld/data"
frames = defaultdict(list)

for folder in os.listdir(directory):
    full_path = os.path.join(directory, folder)
    for filename in os.listdir(full_path):
        if ".soln" in filename:

            file = open(f"{full_path}/{filename}", 'rb')
            object_file = pickle.load(file)
            # for key, value in object_file['solution'].items():
            # object_file['solution'][key] = value
            del object_file['solution'][0]
            df2 = pd.DataFrame(object_file['solution'], index=list(range(len(object_file['solution']))),
                               columns=['action'])
            if len(df2) == 0:
                pass
            if len(df2) != 0:
                frames[folder].append(df2)

for key in frames:
    shuffle(frames[key])


def loadData(df, testing=False, gap=6):
    colsToDrop = []
    for col in df.keys():
        if len(df[col].unique()) < 1:
            colsToDrop.append(col)
    df = df.drop(columns=colsToDrop)

    # df = df.loc[[key for key, row in (df.shift() == df).iterrows() if not all(row)]]

    N = gap

    if not testing:
        df.loc[-1] = ["null" for _ in df.keys()]

    delete = []
    for index, row in df.iterrows():
        pass
        # if pd.isnull(row['Act_B']) and pd.isnull( row['Act_A']):
        # delete.append(index)
    df = df.drop(delete)
    df = df.groupby(list(df.keys())).head(float('inf'))
    df.reset_index(drop=True, inplace=True)

    if not testing:

        s1, s2 = [], []
        for index, row in df.iterrows():
            if index > len(df) - N - 2:
                break
            for i in range(N):
                s1.append(index)
                s2.append(index + i + 1)

        # This feature is added to ensure domain of time slices are equal
        # However with 5k episodes we should be able to ensure that all outomes are included
        # We will have to find out

        # make sure not to add the null state more than once so we subtract one from the length
        s1 += list(range(len(df) - N - 1, len(df) - 1))
        s2 += [-1 for i in range(len(df) - N - 1, len(df) - 1)]

        dataframeleft = df.iloc[s1]
        dataframeright = df.iloc[s2]
    else:
        dataframeleft = df
        dataframeright = df
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

    keysToRename = {}
    for k, r in combined.iterrows():
        for col in combined.keys():
            if "-" in col:
                keysToRename[col] = col.replace("-", "X")
            if isinstance(r[col], str):
                # r[col] = ''.join([i for i in r[col] if not i.isdigit()])
                r[col] = r[col].strip(punctuation)
                r[col] = r[col].replace("-", "X")
                """if not r[col] in found:
                    rstring = ''.join(random.choice(letters) for i in range(5))
                    while rstring in found.values():
                        rstring = ''.join(random.choice(letters) for i in range(5))
                    found[r[col]] = rstring"""
                # r[col] = found[r[col]]

    combined = combined.rename(keysToRename, axis=1)  #
    return combined


def processSolutions(frame, f_key, gap):

    training_dataset, test_dataset = train_test_split(frame[:(len(frame)//2)])

    solutions = []
    for f in training_dataset:
        # Make sure there are more rows than gap, otherwise it will be incomplete
        if len(f) > gap:
            solutions.append(loadData(f, False, gap))

    test_dataset = pd.concat(test_dataset, axis=0, ignore_index=True)
    training_dataset = pd.concat(solutions, axis=0, ignore_index=True)

    convertToString(loadData(training_dataset, True, gap), f"{output_folder}/train_data_{f_key}")
    convertToString(loadData(test_dataset, False, gap), f"{output_folder}/test_data_{f_key}")
    #f"{output_folder}/test_data_gap{gap}_{f_key}.txt", sep = ",", index = False


def convertToString(data, file_name):
    output = ""
    for frame in data:
        if len(frame) == 0:
            print("test")
        output += " ".join([l.strip("()").upper() for l in list(frame["action"].to_numpy())])
        output += "\n"

    text_file = open(file_name, "w")
    text_file.write(output)
    text_file.close()

def runThroughFrames():
    for key, f in frames.items():
        print("Data Type:" + key)

        for gap in range(2, 7):
            print("Data Gap:" + str(gap))

            if not os.path.isfile( f"{output_folder}/train_data_gap{gap}_{key}.txt"):
                processSolutions(f, key, gap)


runThroughFrames()
