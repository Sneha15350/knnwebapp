from sklearn.datasets import load_iris #loads the dataset
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
#import pickle
st.title('IRIS CLASSIFIER USING KNN')
st.write('Streamlit app for KNN iris dataset')
var=load_iris()
x=var.data #input
y=var.target #output
xmin=np.min(x,axis=0)
xmax=np.max(x,axis=0)
model=KNeighborsClassifier(n_neighbors=13,metric='euclidean')
model.fit(x,y)
sepal_length=st.slider('Sepal length: ',float(xmin[0]),float(xmax[0]))
sepal_width=st.slider('Sepal width: ', float(xmin[1]),float(xmax[1]))
petal_length=st.slider('Petal length: ',float(xmin[2]),float(xmax[2]))
petal_width=st.slider('Petal width: ', float(xmin[3]),float(xmax[3]))
y_pred=model.predict([[sepal_length,sepal_width,petal_length,petal_width]])
op=['Setosa','Versicolor','Verginica']
st.title(op[y_pred[0]])
