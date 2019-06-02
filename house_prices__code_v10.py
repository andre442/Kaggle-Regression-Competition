import pandas as pd
import math
import numpy as np
from sklearn.metrics import mean_absolute_error
train_file_path = 'C:\\Users\\Andre\\Desktop\\competicao kaggle\\train.csv'
test_data_path = 'C:\\Users\\Andre\\Desktop\\competicao kaggle\\test.csv'
#########################################################################
#importando bases de dados
base_tr = pd.read_csv(train_file_path)
#################################################
#verificando atributos com valores nulos
r=0
ss=base_tr.isnull().sum().to_frame('nulls')
nulos = ss[ss.nulls>0]
nulas = []
for r in range(0,len(nulos)):
    nulas.append(nulos.index[r])
################################################
# inicio do pre processamento dos dados
############################################################### deletando outliers e fazendo replace de nan para medianas
#base_tr['SalePrice']=np.log(base_tr['SalePrice'])  
base_tr = base_tr[base_tr.GrLivArea < 4500]
base_tr['GrLivArea'] = np.log(base_tr['GrLivArea'])
#base_tr['GarageYrBlt'].fillna(base_tr['GarageYrBlt'].median(),inplace=True)
base_tr['MasVnrArea'].fillna(base_tr['MasVnrArea'].median(),inplace=True)
base_tr['TotalBsmtSF'].fillna(base_tr['TotalBsmtSF'].median(),inplace=True)
base_tr['LotFrontage'].fillna(base_tr['LotFrontage'].median(),inplace=True)
################################################################ transformando com labelencoder variáveis catecóricas que estavam numéricas
base_tr['MSSubClass'] = base_tr['MSSubClass'].apply(str)
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
######################################################################## preenchendo valores NAN de cat com 'None'
for i in range(1,80):
    if base_tr[base_tr.columns[i]].describe().dtype == 'O' and base_tr.columns[i:(i+1)] in nulas:
        base_tr[base_tr.columns[i]].fillna('None',inplace=True)
i=0
######################################################################## preenchendo valores NAN de numericas com valor mais frequente
for i in range(1,80):
    if base_tr[base_tr.columns[i]].describe().dtype == 'float64' and base_tr.columns[i:(i+1)] in nulas:  
        base_tr[base_tr.columns[i]].fillna(base_tr[base_tr.columns[i]].value_counts().idxmax(),inplace=True)
i=0
##################################################################### separando atributos previsores e atritubo classe
previsores = base_tr.iloc[:, 1:80].values
classe = base_tr.iloc[:, 80].values
################################################################ importando labelencoder para transformar variaveis categóricas em numéricas   
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
################################################################ loop para verificar as variaveis categoricas e fazer a transformacao
vcategoricas = []
for a in range(0,len(previsores[0,:])):
    if type(previsores[a,a])==str:
        vcategoricas.append(a)
        previsores[:, a] = labelencoder_previsores.fit_transform(previsores[:, a])
a=0
################################################################ aplicando one hot encoder        
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = vcategoricas)
#previsores = onehotencoder.fit_transform(previsores).toarray()
############################################################################        
classe = classe.reshape(-1,1)
#aplicando padronização standardscaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
scaler2 = StandardScaler()
classe = scaler2.fit_transform(classe)
##############################################################
# fim do pre processamento dos dados
##############################################################
##############################################################
# criacao e treinamento do modelo
##############################################################
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe,
                                                                  test_size = 0.0005,
                                                                  random_state = 0)
import xgboost
regressor=xgboost.XGBRegressor(n_estimators=800, max_depth=4, learning_rate=0.09)
regressor.fit(X_treinamento,y_treinamento)

score_treinamento = regressor.score(X_treinamento, y_treinamento)
score_teste = regressor.score(X_teste, y_teste)
#############################################################
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






#################################################
#importando bases de dados de teste 
#################################################
base_tr = pd.read_csv(test_data_path)
#################################################
# inicio do pre processamento dos dados
#verificando atributos com valores nulos
r=0
ss=base_tr.isnull().sum().to_frame('nulls')
nulos = ss[ss.nulls>0]
nulas = []
for r in range(0,len(nulos)):
    nulas.append(nulos.index[r])
################################################
# inicio do pre processamento dos dados
############################################################### deletando outliers e fazendo replace de nan para medianas
#base_tr['SalePrice']=np.log(base_tr['SalePrice'])  
#base_tr = base_tr[base_tr.GrLivArea < 4500]
base_tr['GrLivArea'] = np.log(base_tr['GrLivArea'])
#base_tr['GarageYrBlt'].fillna(base_tr['GarageYrBlt'].median(),inplace=True)
base_tr['MasVnrArea'].fillna(base_tr['MasVnrArea'].median(),inplace=True)
base_tr['TotalBsmtSF'].fillna(base_tr['TotalBsmtSF'].median(),inplace=True)
base_tr['LotFrontage'].fillna(base_tr['LotFrontage'].median(),inplace=True)
################################################################ transformando com labelencoder variáveis catecóricas que estavam numéricas
base_tr['MSSubClass'] = base_tr['MSSubClass'].apply(str)
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
######################################################################## preenchendo valores NAN de cat com 'None'
for i in range(1,80):
    if base_tr[base_tr.columns[i]].describe().dtype == 'O' and base_tr.columns[i:(i+1)] in nulas:
        base_tr[base_tr.columns[i]].fillna('None',inplace=True)
i=0
######################################################################## preenchendo valores NAN de numericas com valor mais frequente
for i in range(1,80):
    if base_tr[base_tr.columns[i]].describe().dtype == 'float64' and base_tr.columns[i:(i+1)] in nulas:  
        base_tr[base_tr.columns[i]].fillna(base_tr[base_tr.columns[i]].value_counts().idxmax(),inplace=True)
i=0
##################################################################### separando atributos previsores e atritubo classe
previsores = base_tr.iloc[:, 1:80].values
#classe = base_tr.iloc[:, 80].values
################################################################ importando labelencoder para transformar variaveis categóricas em numéricas   
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
################################################################ loop para verificar as variaveis categoricas e fazer a transformacao
vcategoricas = []
for a in range(0,len(previsores[0,:])):
    if type(previsores[a,a])==str:
        vcategoricas.append(a)
        previsores[:, a] = labelencoder_previsores.fit_transform(previsores[:, a])
a=0
################################################################ aplicando one hot encoder        
#from sklearn.preprocessing import OneHotEncoder
#onehotencoder = OneHotEncoder(categorical_features = vcategoricas)
#previsores = onehotencoder.fit_transform(previsores).toarray()
################################################################aplicando padronização standardscaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
##############################################################
# fim do pre processamento dos dados
##############################################################
base_final = pd.read_csv(train_file_path)
classe2 = base_final.iloc[:, 80].values
classe2 = classe2.reshape(-1,1)
scaler_classe = StandardScaler()
classe2 = scaler_classe.fit_transform(classe2)
preds = regressor.predict(previsores)
preds = scaler_classe.inverse_transform(preds)
preds= preds.ravel()  #convertendo para 1 dim
################################################################
###gerando arquivo final
################################################################
my_submission = pd.DataFrame({'Id': base_tr.Id, 'SalePrice': preds})
my_submission.to_csv('submission11.csv', index=False)
