
import pandas as pd
import numpy as np
import zipfile
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import pickle

# paramiters
n_estimators=200
max_depth=10
min_samples_leaf=3

output_file = 'model.bin'

# data preperation 
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall('')

df_marvel = pd.read_csv('marvel-wikia-data.csv')
df_dc = pd.read_csv('dc-wikia-data.csv')

df_marvel.rename(columns={'Year': 'YEAR'}, inplace=True)
df_full = pd.concat([df_marvel, df_dc], axis=0, ignore_index=True)

df_full.columns = df_full.columns.str.lower().str.replace(' ', '_')
df_full = df_full.drop(['gsm','urlslug','year','page_id'], axis=1)

def parse_dates(date):
    if isinstance(date, str):  # Check if the value is a string
        try:
            # Handle the 'Sep-64' format
            if '-' in date:  
                month, year = date.split('-')
                
                # Convert the year based on assumed century
                if int(year) < 30:   # Assume 21st century for years like '00' - '39'
                    year = '20' + year
                else:                # Assume 20th century for years '40' - '99'
                    year = '19' + year
                    
                return pd.to_datetime(f'{month}-{year}', format='%b-%Y')
            
            # Handle the '1971, June' type format
            else:
                return pd.to_datetime(date, format='%Y, %B')
                
        except ValueError:
            return pd.NaT 
    return pd.NaT

df_full['first_appearance'] = df_full['first_appearance'].apply(parse_dates)


values_to_drop_from_target = [
    'Agender Characters', 
    'Genderless Characters', 
    'Genderfluid Characters', 
    'Transgender Characters'
]

df_full = df_full.dropna(thresh=7)
df_full = df_full[~df_full['sex'].isin(values_to_drop_from_target)]
df_full = df_full.dropna(subset=['sex'])


numerical = ['appearances']
categorical = ['id','align','eye','hair','alive']


df_full[numerical] = df_full[numerical].fillna(0)
df_full[categorical] = df_full[categorical].apply(lambda x: x.fillna(x.mode()[0]))

df_full.sex = (df_full.sex == 'Female Characters').astype(int)


# Split and train the model 
print('Starting split and train the model')
df_train, df_test = train_test_split(df_full, test_size=0.2, random_state=1)

def train(df_train, y_train):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)


    model = RandomForestClassifier(n_estimators=n_estimators,
                        max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        random_state=1)    
    model.fit(X_train, y_train)
    
    return dv, model

# training the final model 
def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

dv, model = train(df_train, df_train.sex.values)
y_pred = predict(df_test, dv, model)

y_test = df_test.sex.values
auc = roc_auc_score(y_test, y_pred)
auc


# save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
print(f'The model is saved to {output_file}')




