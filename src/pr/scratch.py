import pickle
import string
from collections import defaultdict

import pandas as pd
import os
from random import shuffle
from sklearn.model_selection import train_test_split
from string import punctuation

directory = "/home/mitch/PycharmProjects/diverse-plan-rec/pyperplan/benchmarks/depot/"
frames = []
for filename in os.listdir(directory):
    if ".soln" in filename:

        file = open(f"{directory}{filename}",'rb')
        object_file = pickle.load(file)
        for key, value in object_file['solution'].items():
            object_file['solution'][key] = "_".join(value)
        object_file['solution'] = [{"action":value} for value in object_file['solution'].values()]
        df = pd.DataFrame(object_file['trace'])
        df2 = pd.DataFrame(object_file['solution'], index=list(range(len(object_file['solution']))))
        horizontal_stack = pd.concat([df, df2], axis=1)
        frames.append(horizontal_stack)

shuffle(frames)

def loadData(df):
    colsToDrop = []



    for col in df.keys():
        if len(df[col].unique()) == 1 or "sub_goal" in col or "Hands" in col or "close" in col:
            colsToDrop.append(col)
        df[col].fillna('null')
    df = df.drop(columns=colsToDrop)

    #df = df.loc[[key for key, row in (df.shift() == df).iterrows() if not all(row)]]

    N = 6

    df.loc[-1] = ["null" for _ in df.keys()]

    delete = []
    for index, row in df.iterrows():
        pass
        #if pd.isnull(row['Act_B']) and pd.isnull( row['Act_A']):
            #delete.append(index)
    df = df.drop(delete)
    df = df.groupby(list(df.keys())).head(float('inf'))
    df.reset_index(drop=True, inplace=True)

    s1, s2 = [-1], [0]
    for index, row in df.iterrows():
        if index > len(df) - N-1:
            break
        for i in range(N):
            s1.append(index)
            s2.append(index + i + 1)

    s1 += list(range(len(df)-1-N, len(df)-1))
    s2 += [-1 for i in range(len(df)-1-N, len(df)-1)]

    dataframeleft = df.iloc[s1]
    dataframeright = df.iloc[s2]

    dataframeright.append(pd.Series(), ignore_index=True)

    # now rename columns
    renameColumns = {}
    for col in df.keys():
        renameColumns[col] = col + "0"
    dataframeleft = dataframeleft.rename(columns=renameColumns)
    dataframeleft.reset_index(drop=True, inplace=True)

    renameColumns = {}
    for col in df.keys():
        renameColumns[col] = col + "1"
    dataframeright = dataframeright.rename(columns=renameColumns)
    dataframeright.reset_index(drop=True, inplace=True)


    combined = pd.concat([dataframeleft, dataframeright], axis=1)

    count = 0
    for key in combined.keys():
        if "Hands" in key:
            combined[key] = combined[key].fillna('Empty')
        elif "close" in key.lower():
            combined[key] = combined[key].fillna('null')
        elif "act" in key.lower():
            combined[key] = combined[key].fillna('none')
            combined.iloc[0][key] = "null"
        else:
            combined[key] = combined[key].fillna('null')


    assert(len(combined) == len(dataframeleft))

    # Make sure domain of each sides aer the same
    for key in dataframeleft.keys():
        assert(len(dataframeleft[key].unique()) == len(dataframeright[key[:-1]+"1"].unique()))


    found = {}
    letters = string.ascii_lowercase
    import random
    for k, r in combined.iterrows():
        for col in combined.keys():
            if isinstance(r[col], str):

                r[col] = ''.join([i for i in r[col] if not i.isdigit()])
                r[col] = r[col].strip(punctuation)
                r[col] = r[col].replace("-","X")
                """if not r[col] in found:
                    rstring = ''.join(random.choice(letters) for i in range(5))
                    while rstring in found.values():
                        rstring = ''.join(random.choice(letters) for i in range(5))
                    found[r[col]] = rstring"""
                #r[col] = found[r[col]]

    return combined




foundItems = defaultdict(set)
indexes = set()
for key, f in enumerate(frames):
    for k in f.keys():
        nextSet = set(f[k].unique())
        if len(nextSet - foundItems[k]) > 0:
            foundItems[k] |= nextSet
            indexes.add(key)
uniqueSet = [frames[i] for i in indexes]

#frames = [frames[i] for i in range(len(frames)) if i not in indexes]
training_dataset, test_dataset = train_test_split(frames)

#training_dataset = training_dataset[:(len(training_dataset) - len(uniqueSet)) // 4]
#test_dataset = test_dataset[:len(test_dataset) // 4]
#training_dataset += uniqueSet

test_dataset = pd.concat(test_dataset, axis=0, ignore_index=True)
training_dataset = pd.concat(training_dataset[:len(training_dataset)//2], axis=0, ignore_index=True)

training_data = loadData(training_dataset)

loadData(training_dataset).to_csv("train_data.txt", sep=",", index=False)
loadData(test_dataset).to_csv("test_data.txt".format(2), sep=",", index=False)

file.close()