import pandas as pd
import matplotlib.pyplot as plt
import hashlib
df = pd.read_csv("/home/mitch/PycharmProjects/diverse-plan-rec/src/pr/output_underlying_plans_noholding_noarms/train_data_gap1_data-4blocks-4ops.txt")
colsToDrop = []
for col in df.keys():
    if col[-1] == '1':  # or "holding" in col:
        colsToDrop.append(col)
df = df.drop(columns=colsToDrop)

new_df = pd.DataFrame(columns=["S"+str(i+1) for i in range(6)])
hash_object = hashlib.md5(b'Hello World')
print(hash_object.hexdigest())
result = []
#for i in range(7,len(df)-6):
    #data = {"S"+str(ii+1):hashlib.md5(df.iloc[i-ii].to_string(header=False, index=False).encode()) for ii in range(7)}
    #result.append(data)
new_df = new_df.append(result,ignore_index=True)

y = df['action0']
x = df.drop(columns=['action0'])

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = df.fillna("false")
X = df.drop(['action0'], axis=1)

le = LabelEncoder()


y = le.fit_transform(df['action0'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

data =[X_train, X_test]
for x in data:
    for key in x.keys():
        x[key] = le.fit_transform(x[key].astype(str))


from xgboost import XGBClassifier

model = XGBClassifier(use_label_encoder=False)
model.fit(X_train, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.feature_importances_
})
importances = importances.sort_values(by='Importance', ascending=False)
plt.gcf().subplots_adjust(bottom=.3)
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importance obtained from coefficients 4 blocks ', size=20)
plt.xticks(rotation='vertical')
plt.show()