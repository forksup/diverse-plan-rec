import math
import os
from collections import defaultdict
import pandas as pd
from math import floor
import numpy as np
noise = .05
output = "/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/output_underlying_plans_noholding_noarms_5%noise/"
if __name__ == '__main__':
    data = defaultdict(lambda: defaultdict(list))
    for dirpath, _, filenames \
            in os.walk('/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/input_data/output_underlying_plans_noholding_noarms'):
        for j, f in enumerate(filenames):
            if "test" in f:
                continue
            df = pd.read_csv(os.path.abspath(os.path.join(dirpath, f)), index_col=False)
            dfupdate = df.sample(floor(len(df)*noise))
            dfupdate.Quantity = 0
            df.update(dfupdate)
            update_list = dfupdate.index.tolist()
            df.to_csv(output+f, index=False, na_rep='null')


