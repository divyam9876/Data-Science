import pandas as pd
from sklearn.model_selection import train_test_split

data_file = '/Users/divyamaddipatla/Desktop/Final/new/input/accident_1.csv'

data = pd.read_csv(data_file, header=0)

data = data[['REGION', 'URBANICITY', 'WEIGHT' ,'MONTH','DAY_WEEK', 'HOUR', 'HARM_EV', 'MAN_COLL', 
             'RELJCT2', 'TYP_INT', 'REL_ROAD', 'LGT_COND', 'WEATHER', 'ALCOHOL', 'MAX_SEV']]

X = data.drop(columns=['MAX_SEV'])
y = data['MAX_SEV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

train_f_data = pd.concat([X_train, y_train], axis=1)
test_f_data = pd.concat([X_test, y_test], axis=1)

train_f_data.to_csv("/Users/divyamaddipatla/Desktop/Final/new/input/train_f_data.csv", index=False)
test_f_data.to_csv("/Users/divyamaddipatla/Desktop/Final/new/input/test_f_data.csv", index=False)

print("Training and test data merged and saved as train_f_data.csv and test_f_data.csv.")

