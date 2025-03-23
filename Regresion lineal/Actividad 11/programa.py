import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler


dataframe = pd.read_csv("./usuarios_win_mac_lin.csv")

dataframe.drop(['clase'], axis=1).hist()

X = np.array(dataframe.drop(['clase'], axis=1))
y = np.array(dataframe['clase'])

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Dimensiones de X:", X.shape)

model = linear_model.LogisticRegression(max_iter=500)
model.fit(X, y)

predictions = model.predict(X)
print("Primeras 5 predicciones:", predictions[:5])
print("Precisión en datos de entrenamiento:", model.score(X, y))

validation_size = 0.20
seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(
    X, y, test_size=validation_size, random_state=seed
)

name = 'LogisticRegression'
kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True) 

cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')

msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions = model.predict(X_validation)
print("Precisión en datos de validación:", accuracy_score(Y_validation, predictions))

# Reporte de resultados
print("Matriz de confusión:\n", confusion_matrix(Y_validation, predictions))
print("Reporte de clasificación:\n", classification_report(Y_validation, predictions))
