from fastapi import FastAPI
import numpy as np
import pandas as pd
import pickle
import uvicorn
import os
from iris import Iris


app=FastAPI()

#loading all the pickle files





# Folder containing your files

with open("models/lr_classifier.pkl","rb") as f:
    lr_classifier=pickle.load(f)

with open("models/svm_classifier.pkl","rb") as f:
    svm_classifier=pickle.load(f)

with open("models/knn_classifier.pkl","rb") as f:
    knn_classifier=pickle.load(f)

with open("models/rf_classifier.pkl","rb") as f:
    rf_classifier=pickle.load(f)


@app.get("/")
async def root():
    return {"message": "Home Page for IRIS"}


@app.post("/predict/lr")
async def LogisticRegression(data: Iris):
    """
    # This is an API for an Iris Species classifier based on their features
    ### It should works sending measurements in a json format like this:
    {\n
        "sepal_length": "float"
        "sepal_width": "float"
        "petal_length": "float"
        "petal_width": "float"
    }\n
    """
    # We predict the type of iris: setosa, versicolor or virginica 

#     data = data.dict()
#     sepal_length=data['sepal_length']
#     sepal_width=data['sepal_width']
#     petal_length=data['petal_length']
#     petal_width=data['petal_width']

#     species = ['setosa', 'versicolor', 'virginica']
#    # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
#     prediction = lr_classifier.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    
#     return {
#         'prediction': + species[int(prediction)]
#     }
    species = ['setosa', 'versicolor', 'virginica']
    pred = lr_classifier.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {'Prediction: ' + species[int(pred)]}

@app.post("/predict/rf")
async def RandomForest(data: Iris):
    """
    # This is an API for an Iris Species classifier based on their features
    ### It should works sending measurements in a json format like this:
    {\n
        "sepal_length": "float"
        "sepal_width": "float"
        "petal_length": "float"
        "petal_width": "float"
    }\n
    """
    # We predict the type of iris: setosa, versicolor or virginica 

    species = ['setosa', 'versicolor', 'virginica']
    pred = rf_classifier.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {'Prediction: ' + species[int(pred)]}


@app.post("/predict/svm")
async def SVM(data: Iris):
    """
    # This is an API for an Iris Species classifier based on their features
    ### It should works sending measurements in a json format like this:
    {\n
        "sepal_length": "float"
        "sepal_width": "float"
        "petal_length": "float"
        "petal_width": "float"
    }\n
    """
    # We predict the type of iris: setosa, versicolor or virginica 

    species = ['setosa', 'versicolor', 'virginica']
    pred = svm_classifier.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {'Prediction: ' + species[int(pred)]}

@app.post("/predict/knn")
async def KNN(data: Iris):
    """
    # This is an API for an Iris Species classifier based on their features
    ### It should works sending measurements in a json format like this:
    {\n
        "sepal_length": "float"
        "sepal_width": "float"
        "petal_length": "float"
        "petal_width": "float"
    }\n
    """
    # We predict the type of iris: setosa, versicolor or virginica 

    species = ['setosa', 'versicolor', 'virginica']
    pred = knn_classifier.predict([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    return {'Prediction: ' + species[int(pred)]}



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)