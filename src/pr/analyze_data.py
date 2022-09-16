import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import re

pickle_path = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/test_domain_fruit_training_data/data_split.obj"

with open(pickle_path, "rb") as input_file:
    data = pickle.load(input_file)


training, test = data
traindata = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/output_underlying_plans_noholding_noarms/train_data_gap1_data-4blocks-4ops.txt"
testdata = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/output_underlying_plans_noholding_noarms/test_data-4blocks-4ops.txt"

if not isinstance(training, list):
    training = [training]


for i in range(len(training)):
    training[i].iloc[-1] = ['end' for _ in training[i].keys()]

combined = pd.concat(training, axis=0, ignore_index=True)

combined = combined.rename(columns = {key : key.replace("-", "X") for key in combined.keys()}).fillna("null")
combined = combined.dropna(how='all')
combined = combined.reindex(columns=sorted(list(combined.keys())))
if "action" in combined.keys():
    combined['action'] = combined['action'].apply(lambda row : re.sub("[()]","", row))

outputpath = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/test_domain_fruit_training_data"
combined.to_csv(f"{outputpath}/markovtraining.txt", na_rep='null', index=False)
columns = set()
for i in range(2):
    for pd in data[i]:
        columns |= {key for key in pd.keys()}

end = -1
endstate = tuple()
all_problems = [set(), set()]
state_count = [defaultdict(lambda: 0), defaultdict(lambda: 0)]
start = set()
end = set()


for i in range(2):
    for pd in data[i]:
        for col in columns-set(pd.keys()):
            pd[col] = np.nan
        df = pd.drop(['action'], axis=1).fillna("")
        df = df.reindex(sorted(df.columns), axis=1)
        s = list(df.iloc[0])
        e = list(df.iloc[-1])
        start.add(tuple(s))
        end.add(tuple(e))
        problem = tuple(s+e)
        all_problems[i].add(problem)
        state_count[i][problem] += 1

test = "asdf"
"""
for key, row in preprocessed_training.iterrows():
    if row[0] == 'end':
        problem = tuple(preprocessed_training.iloc[end+1].values+row.values)
        state_count[problem] += 1
        all_problems.add(problem)
        end = key
done = "done" """