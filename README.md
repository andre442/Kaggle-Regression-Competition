# Kaggle-Regression-Competition

Complete algorithm that I developed for the Kaggle competition to predict the price of houses 
using regression methods with machine learning. With 79 explanatory variables describing (almost) 
every aspect of residential homes in Ames, Iowa, this competition challenges to predict the final price of each home.
This following code achieved a performance among the  9% best results of 5933 teams that joined  the competition!

Beginning importing the main libraries and training data:

```
import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
train_file_path = 'C:\\Users\\Andre\\Desktop\\competicao kaggle\\train.csv'
#importando bases de dados
base_tr = pd.read_csv(train_file_path)
```
First of all, searching for null atributes:

```
r=0
ss=base_tr.isnull().sum().to_frame('nulls')
nulos = ss[ss.nulls>0]
nulas = []
for r in range(0,len(nulos)):
    nulas.append(nulos.index[r])
```
![nulos](https://user-images.githubusercontent.com/50015049/58761032-c48ea180-8515-11e9-96df-94b171132a54.png)

Some of attributes have outliers, and NaN in the data. Now is time to adjust some categorical values by changing string names and using median
of null values on numerical features:

```
base_tr = base_tr[base_tr.GrLivArea < 4500]                                      #removendo outliers

base_tr['GrLivArea'] = np.log(base_tr['GrLivArea'])                              #obtendo linearidade nos extremos

base_tr['MasVnrArea'].fillna(base_tr['MasVnrArea'].median(),inplace=True)        #adicionando mediana em valores nulos
base_tr['TotalBsmtSF'].fillna(base_tr['TotalBsmtSF'].median(),inplace=True)
base_tr['LotFrontage'].fillna(base_tr['LotFrontage'].median(),inplace=True)
base_tr['MSSubClass'] = base_tr['MSSubClass'].apply(str)                        #alterando nomes em variáveis categóricas
base_tr['YrSold'] = base_tr['YrSold'].astype(str)
base_tr['MoSold'] = base_tr['MoSold'].astype(str)
base_tr['Functional'] = base_tr['Functional'].fillna('Typ')
base_tr['Electrical'] = base_tr['Electrical'].fillna("SBrkr")
base_tr['KitchenQual'] = base_tr['KitchenQual'].fillna("TA")
base_tr['Exterior1st'] = base_tr['Exterior1st'].fillna(base_tr['Exterior1st'].mode()[0])
base_tr['Exterior2nd'] = base_tr['Exterior2nd'].fillna(base_tr['Exterior2nd'].mode()[0])
base_tr['SaleType'] = base_tr['SaleType'].fillna(base_tr['SaleType'].mode()[0])
base_tr["PoolQC"] = base_tr["PoolQC"].fillna("None")
for col in ('GarageArea', 'GarageCars'):
    base_tr[col] = base_tr[col].fillna(0)
```

Creating a loop to fill some null values from others features with the string 'O' and numerical features with the most frequent value:
```
for i in range(1,80):
    if base_tr[base_tr.columns[i]].describe().dtype == 'O' and base_tr.columns[i:(i+1)] in nulas:
        base_tr[base_tr.columns[i]].fillna('None',inplace=True)
i=0

for i in range(1,80):
    if base_tr[base_tr.columns[i]].describe().dtype == 'float64' and base_tr.columns[i:(i+1)] in nulas:  
        base_tr[base_tr.columns[i]].fillna(base_tr[base_tr.columns[i]].value_counts().idxmax(),inplace=True)
i=0
```

Now time to split in to target(class) and features(previsores) :
```
previsores = base_tr.iloc[:, 1:80].values
classe = base_tr.iloc[:, 80].values
```
For last, using label_encoder and standard_scaler to finish the pre-processing phase:
```
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()

#just a loop for mark all categorical variables in a array
vcategoricas = []
for a in range(0,len(previsores[0,:])):
    if type(previsores[a,a])==str:
        vcategoricas.append(a)
        previsores[:, a] = labelencoder_previsores.fit_transform(previsores[:, a])
a=0

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
scaler2 = StandardScaler()
classe = scaler2.fit_transform(classe)
```

Creating a model and training for test:
```
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe,
                                                                  test_size = 0.25,
                                                                  random_state = 0)
import xgboost
regressor=xgboost.XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.09)
regressor.fit(X_treinamento,y_treinamento)

score_treinamento = regressor.score(X_treinamento, y_treinamento)
score_teste = regressor.score(X_teste, y_teste)
```
Adjusting the model parameters, i´ve find that these values are the best for reach the right balance between accuracy and overfitting.
The score for training and test was respectively:
```
#score_trein 0.9270571423899016
#score_teste 0.999572850099914
```
Testing the predictions:
```
previsoes = regressor.predict(X_teste)
y_teste   = scaler2.inverse_transform(y_teste)
previsoes = scaler2.inverse_transform(previsoes)
previsoes = previsoes.reshape(-1,1)
#Função para calcular (RMSLE)
def rmsle(y, y_pred):
	assert len(y) == len(y_pred)
	terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
	return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5
R = rmsle(y_teste, previsoes)
mae = mean_absolute_error(previsoes, y_teste)
```

A mean absolute error of 15475.60602525685 on the test group (0.25 of data) and RMSLE = 0.11595850765352847
For better accuracy, you will need explore more aspects of the features variables and make a more deep analysis to obtain better results.
The remaining code just apply the same pre-processing to the test data and generates at the end a submission file.
 




