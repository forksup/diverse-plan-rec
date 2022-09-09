
from collections import defaultdict
import pandas as pd
import os
from random import shuffle, choices, random
from sklearn.model_selection import train_test_split
from string import punctuation
import pickle

noise = 0.00

def convert_gap_one_test():
    gap1_file = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/output_underlying_plans_noholding_noarms/train_data_gap1_data-6blocks-4ops.txt"
    df = pd.read_csv(gap1_file)

    for key in df.keys():
        if key[-1] != 1:
            df[key[:-1]] = df[key]
            df = df.drop(columns=[key])
    df.to_csv("/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/output_underlying_plans_noholding_noarms/train_test.txt",index=False,na_rep='null')

output_folder = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/output_underlying_plans_noholding_noarms_CAP"
directory = "/mnt/watchandhelp/PycharmProjects/pddl-generators/blocksworld/data"
frames = defaultdict(list)

"""
for filename in os.listdir(directory):
    df = pd.read_csv(directory + "/" + filename, skipinitialspace=True)
    # Originally I added the init state here but we will ignore that and moving forward not add this first state
    if df.iloc[0]['Act_A'] == numpy.nan:
        df = df.drop(0)
    frames['loadtable'].append(df)
"""
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
            df1 = pd.DataFrame(object_file['trace'], index=list(range(len(object_file['trace']))))

            if len(df2) == 0:
                pass
            if len(df2) != 0:
                frames[folder].append(pd.concat([df1, df2], axis=1))
            file.close()

"""
for folder in os.listdir(directory):
    full_path = os.path.join(directory, folder)
    for filename in os.listdir(full_path):
        if ".soln" in filename:

            file = open(f"{full_path}/{filename}",'rb')
            object_file = pickle.load(file)
            #for key, value in object_file['solution'].items():
                #object_file['solution'][key] = value
            df = pd.DataFrame(object_file['trace'])
            df2 = pd.DataFrame(object_file['solution'], index=list(range(len(object_file['solution']))),columns=['action'])
            horizontal_stack = pd.concat([df, df2], axis=1)
            if len(horizontal_stack) != 1:
                frames[folder].append(horizontal_stack)
"""

for key in frames:
    shuffle(frames[key])


def load_data(df, testing=False, gap=6):
    if len(df) < 4:
        return pd.DataFrame()

    colsToDrop = []
    for col in df.keys():
        if len(df[col].unique()) < 1 or "holding" in col or "arm" in col:
            colsToDrop.append(col)
    df = df.drop(columns=colsToDrop)

    df = df.groupby(list(df.keys())).head(float('inf'))
    df.reset_index(drop=True, inplace=True)

    if testing:
        return rename_columns(df)

    s1, s2 = [], []

    for index, row in df.iterrows():
        if index == (len(df) - gap):
            break
        # FOR CAP
        for i in range(index+1, index + gap+1):
            s1.append(index)
            s2.append(i)

    dataframeleft = df.iloc[s1]
    dataframeright = df.iloc[s2]

    dataframeright.append(pd.Series(), ignore_index=True)

    # now rename columns
    dataframeleft = dataframeleft.rename(columns={col: str(col) + "0" for col in df.keys()}).reset_index()
    dataframeright = dataframeright.rename(columns={col: str(col) + "1" for col in df.keys()}).reset_index()

    # The problem is, is that when you concat multiple datafrmes there are columns which do not exist in the other.
    # So we fill those with various null values
    combined = pd.concat([dataframeleft, dataframeright], axis=1)
    combined.reset_index(drop=True, inplace=True)
    combined = combined.replace(r'^\s*$', "null", regex=True)

    assert len(combined) == len(dataframeleft)

    return rename_columns(combined)

def rename_columns(combined):
    keysToRename = {}
    for col in combined.keys():
        if "-" in col:
            keysToRename[col] = col.replace("-", "X")
        for k, r in combined.iterrows():
            if isinstance(r[col], str):
                r[col] = r[col].strip(punctuation)
                r[col] = r[col].replace("-", "X")

    return combined.rename(keysToRename, axis=1)  #

def process_solutions(frames, f_key):
    split = train_test_split(frames)
    input_file = open(r'data_split.obj', 'wb')
    #split = pickle.load(input_file)
    pickle.dump(split, input_file)

    preprocessed_training, test_dataset = split

    for f in range(len(test_dataset)):
        # add seperator to end of episodes
        df2 = {key: 'end' for key in test_dataset[f].keys()}
        test_dataset[f] = test_dataset[f].append(df2, ignore_index=True)
    test_dataset = pd.concat(test_dataset, axis=0, ignore_index=True)

    # For rows with "end" in, change all values to end
    for key, row in test_dataset.iterrows():
        if (row == "end").any():
            for col in test_dataset.keys():
                test_dataset.at[key, col] = 'end'
    test_dataset = fillNull(test_dataset)
    test_dataset = load_data(test_dataset, True)
    test_dataset.to_csv(f"{output_folder}/test_{f_key}.txt", sep=",", index=False)

    for gap in range(1, 5):
        print("Data Gap:" + str(gap))

        solutions = []
        for f in preprocessed_training:
            # Make sure there are more rows than gap, otherwise it will be incomplete
            solutions.append(load_data(f, False, gap))

        training_dataset = pd.concat(solutions, axis=0, ignore_index=True)
        training_dataset = fillNull(training_dataset)
        training_dataset = training_dataset.append({}, ignore_index=True)

        for key0 in training_dataset.keys():
            if "0" in key0:
                key1 = key0[:-1] + "1"
                if len(training_dataset[key0].unique()) != len(training_dataset[key1].unique()):
                    v0 = set(training_dataset[key0].unique())
                    v1 = set(training_dataset[key1].unique())

                    values_not_in_1 = v0 - v1
                    values_not_in_0 = v1 - v0

                    for val in values_not_in_0:
                        training_dataset = training_dataset.append({key0 : val}, ignore_index=True)

                    for val in values_not_in_1:
                        training_dataset = training_dataset.append({key1: val}, ignore_index=True)

        #var = {random.randint(0, len(training_dataset) - 1) for _ in range(math.floor(len(training_dataset) * noise))}

        # say we have n transitions
        # we will reorganize noise / 3 items



            """            import math
        indexes_to_remove = np.random.randint(0, len(training_dataset)-1, size=noise_count)
        rows_to_reinsert = training_dataset.iloc[indexes_to_remove]
        training_dataset = training_dataset.drop(labels=indexes_to_remove, axis=0)
        [training_dataset for key in np.random.randint(0, len(training_dataset)-1, size=noise_count)]"""


        #n = math.floor(len(training_dataset) * noise)
        #rand_idx = np.random.randint(0, len(training_dataset), size=n)
        #n = 0
        #rows = []
        #for _ in range(n):
            #first_slice = training_dataset.sample().iloc[:, :len(training_dataset.keys()) // 2]
            #second_slide =training_dataset.sample().iloc[:, len(training_dataset.keys()) // 2:]
            #rows.append(first_slice.append(second_slide, ignore_index="True"))
        # Append 'n' rows of zeroes to D1
        #training_dataset = training_dataset.append(pd.DataFrame(rows, columns=training_dataset.columns, dtype=int), ignore_index=True)

        # Insert n zeroes into random indices and assign back to column 'b'
        #D2['b'] = np.insert(D1['b'].values, rand_idx, 0)

        training_dataset = training_dataset.fillna("null")
        # Make sure domain of each sides are the same
        for key in training_dataset.keys():
            if "0" in key:
                assert (len(training_dataset[key].unique()) == len(training_dataset[key[:-1] + "1"].unique()))


        training_dataset.to_csv(f"{output_folder}/train_data_gap{gap}_{f_key}.txt", sep=",", index=False)

        for key in training_dataset.keys():
            #Subtract one for end in test dataset
            assert(len(test_dataset[key[:-1]].unique())-1 <= len(training_dataset[key].unique()))


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
    for key, f in frames.items():
        process_solutions(f, key)


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
