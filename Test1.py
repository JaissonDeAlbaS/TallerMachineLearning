# -*- coding: utf-8 -*-
"""
@author: Jaisson De Alba Santos
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
simplefilter(action='ignore', category=FutureWarning)

url = 'Dataset/diabetes.csv'
data = pd.read_csv(url)

# Tratamiento de la data

data.drop(['Pregnancies', 'SkinThickness','DiabetesPedigreeFunction' ], axis= 1, inplace = True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100] #Rango para las edades
nombres = ['1', '2', '3', '4', '5', '6', '7'] #Nombre para los rangos
data.Age = pd.cut(data.Age, rangos, labels=nombres) #Transformalo en forma de rango
data.dropna(axis=0,how='any', inplace=True) #Borra todo lo que sea NAN

# partir la data en dos
#data.info()
#768 / 2

data_train = data[:384] #Va desde el origen hasta 384
data_test = data[384:] #Va desde 384 hasta el final


#x = Toda la data que no tenga la clasificacion
#y = Clasificacion

x = np.array(data_train.drop(['Outcome'], 1))
y = np.array(data_train.Outcome)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3) #Proporcion del 30%

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)

# Regresión Logística

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600) #hiperparametros

# Entreno el modelo
logreg.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


# MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entreno el modelo
svc.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


# ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entreno el modelo
arbol.fit(x_train, y_train)

# MÉTRICAS

print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')
