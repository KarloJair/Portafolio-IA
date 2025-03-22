# Imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#cargamos los datos de entrada
data = pd.read_csv("./articulos_ml.csv")

filtered_data=data[(data['Word count']<=3500)&(data['# Shares']<=80000)]

f1=filtered_data['Word count'].values
f2=filtered_data['# Shares'].values

dataX = filtered_data[["Word count"]]
X_train=np.array(dataX)
y_train=filtered_data['# Shares'].values
 
#Creamoselobjetode RegresiónLinear
regr=linear_model.LinearRegression()
 
  #Entrenamosnuestromodelo
regr.fit(X_train,y_train)

#Hacemoslasprediccionesqueen definitivauna línea(enestecaso,alser2D)
y_pred=regr.predict(X_train)
#Veamosloscoeficienetesobtenidos,Ennuestro caso,seránlaTangente
print('Coefficients:\n',regr.coef_)
#Esteeselvalordonde cortael ejeY(enX=0)
print('Independent term:\n',regr.intercept_)
#ErrorCuadradoMedio
print("Mean squared error:%.2f"%mean_squared_error(y_train,y_pred))
#PuntajedeVarianza. Elmejorpuntajeesun1.0
print('Variance score:%.2f'%r2_score(y_train,y_pred))