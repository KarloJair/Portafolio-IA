# Imports necesarios
import numpy as np
import pandas as pd
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
 
regr=linear_model.LinearRegression()
 
regr.fit(X_train,y_train)

#Hacemos las predicciones que en definitiva una línea(en este caso,alser2D)
y_pred=regr.predict(X_train)
#Veamos los coeficienetes obtenidos,En nuestro caso,serán la Tangente
print('Coefficients:\n',regr.coef_)
#Este es el valor donde corta el ejeY(enX=0)
print('Independent term:\n',regr.intercept_)
#Error Cuadrado Medio
print("Mean squared error:%.2f"%mean_squared_error(y_train,y_pred))
#Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score:%.2f'%r2_score(y_train,y_pred))