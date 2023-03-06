# IrisFastAPI
This project uses Iris dataset for classification and uses FastAPI as endpoint

- "data" folder contains the IRIS dataset in csv format
- "models" folder contains the pickle files of Random Forest, SVM, KNN and logistic regression classifier models
- iris.py contains IRIS class and its properties
- main.py contains the code for FAST API app
- modelimplementation.ipynb is the python notebook which has all the code for basic EDA and building out models
To run this project, enter the command in Anaconda command line window: uvicorn main:app --reload 
