import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import export_graphviz
#import graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#
# CARGAR DATASET
#
data_train = pd.read_csv(filepath_or_buffer="dataset/datatraining.txt")

# Es necesario hacer una pequeña conversión en el campo date, ya que por las particularidades del dataset toma el dato como tipo object en lugar de date,
# despues debe convertirse a un campo de tipo numerico.
#
data_train["date"] = pd.to_numeric(pd.to_datetime(data_train["date"]))
# print(data_train.date)
#data_test = pd.read_csv(filepath_or_buffer="dataset/datatest.txt")

data_test = pd.read_csv(filepath_or_buffer="dataset/datatest2.txt")
data_test["date"] = pd.to_numeric(pd.to_datetime(data_test["date"]))

print("Entrenamiento")
print(data_train.shape)
print("Prueba")
print(data_test.shape)

#
# PREPARAR DATASET
#

# datos de entrenamiento
array_train = data_train.values
# este parece ser el unico modelo al que la fecha no le genera ruido excesivo, sin embargo las predicciones sin este parametro son ligeramente mejores, por lo tanto podemos afirmar que sigue añadiendo ruido
# por lo tanto se omitira de nuevo.
train = array_train[:, 1:6]
labels = array_train[:, 6]
# es necesario convertir el arreglo de y a entero, ya que presenta problemas con el tipado y lo toma como un tipo unknown
labels = labels.astype('int')
# datos de prueba
array_test = data_test.values
test = array_test[:, 1:6]
label_test = array_test[:, 6]
label_test = label_test.astype('int')

# Normalizar dataset
scaler = MinMaxScaler()
scaler.fit_transform(train)

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X=train, y=labels)
prediction = clf.predict(test)
acc = accuracy_score(label_test, prediction)
print("Precisión Tree", acc)
print(classification_report(label_test, prediction, labels=np.unique(prediction)))
print(confusion_matrix(label_test, prediction))
