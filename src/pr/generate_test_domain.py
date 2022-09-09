# - States are dependent
from collections import defaultdict
import random
import time
from numpy.random import randint
import pandas as pd
import numpy as np
randint(0, [5], 100)

increase = .55

n = 500000


class probability:
    prob = .3

def decision(probability):
    random.seed(time.process_time())
    return random.uniform(0,1) < probability


# slice first, then variable name
probabilities = defaultdict(lambda: defaultdict(probability))

edges = {tuple(["apple", "pineapple"]): 3, tuple(["pineapple", "orange"]): 1, tuple(["orange", "potato"]): 2}

variables = ["apple", "potato", "pineapple", "orange"]
variableValues = [decision(.5) for _ in range(len(variables))]

output = [variableValues]

for i in range(0,n):
    values = {variables[j]: variableValues[j] for j in range(len(variables))}

    for key, slice in edges.items():
        if values[key[0]]:
            probabilities[slice+i][key[1]].prob += increase

    for key, value in enumerate(variables):
        variableValues[key] = decision(probabilities[i+1][value].prob)

    output.append(variableValues[:])

pd.DataFrame(data=output, columns=variables).to_csv("test_output.txt",index=False)


df = pd.DataFrame(data=output, columns=variables)
unique_states = df.drop_duplicates()

unique_states = unique_states.reset_index(drop=True)

state_index = defaultdict(lambda: -1)
for key, row in unique_states.iterrows():
    state_index[hash(tuple(row))] = int(key)
converted_df = df.apply(lambda x: state_index[hash(tuple(x))], axis=1).to_frame()
converted_df.columns = ['state']
from fastparquet import write
write('/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/test_domain_fruit_training_data/outfile.parq', converted_df,compression="UNCOMPRESSED")
#%%
