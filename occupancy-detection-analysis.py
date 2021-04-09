import pandas as pd
import numpy as np
from scipy.stats import shapiro
from scipy.stats import normaltest
import io

dataframe = pd.read_csv(filepath_or_buffer="dataset/datatest.txt")
dataframe["date"] = pd.to_numeric(pd.to_datetime(dataframe["date"]))
columns=dataframe.columns.values
normal=[]
noNormal=[]
"""
Si el valor p ≤ 0.05, entonces rechazamos la hipótesis nula, es decir, asumimos que la distribución de nuestra variable no es normal / gaussiana.
Si el valor p> 0.05, entonces no rechazamos la hipótesis nula, es decir, asumimos que la distribución de nuestra variable es normal / gaussiana.
"""
print("----------Shapiro------------")
for currentColumn in columns:
  data=dataframe[currentColumn]
  stat,p=shapiro(data)
  if p>0.05:
    print("",currentColumn," - ", p)
    normal.append(currentColumn)
  else:
    print("",currentColumn," - ", p)
    noNormal.append(currentColumn)
print("Con distribucion normal: ",normal)
print("Sin distribucion normal: ",noNormal)

normal=[]
noNormal=[]
print("-------------- Normal test ------------------")
for currentColumn in columns:
  data=dataframe[currentColumn]
  stat,p=normaltest(data)
  if p>0.05:
    print("",currentColumn," - ", p)
    normal.append(currentColumn)
  else:
    print("",currentColumn," - ", p)
    noNormal.append(currentColumn)
print("Con distribucion normal: ",normal)
print("Sin distribucion normal: ",noNormal)