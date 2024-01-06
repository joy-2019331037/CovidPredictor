from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


def HomePage(req):
    return render(req,"home.html")

def PredictPage(req):
    return render(req,"predict.html")

def ResultPage(req):
    data = pd.read_csv("C:/Users/User/Desktop/django/CovidPrediction/covid-19 symptoms dataset.csv")
    X=data.drop("infectionProb", axis=1)
    Y=data["infectionProb"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=45)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train,Y_train)

    #fetching user provided values
    var1=int(req.GET['n1'])
    var2=int(req.GET['n2'])
    var3=int(req.GET['n3'])
    var4=int(req.GET['n4'])
    var5=int(req.GET['n5'])

    prediction = model.predict(np.array([[var1,var2,var3,var4,var5]]))

    result ="Patient is Covid Negative"
    prediction_result_color = "green"
    if prediction ==[1]:
        result="Patient is Covid Positive"
        prediction_result_color = "red"
    return render(req,"predict.html",{"result": result,"prediction_result_color": prediction_result_color})