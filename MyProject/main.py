# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:08:14 2022

@author: Engineer Sameer
"""
# pip install fastapi uvicorn

# 1.Library imports
import uvicorn # ASGI (Asynchronous Server Gateway Interface)
from fastapi import FastAPI
from Bank import BankLoan
import pickle
pickle_in=open('StackModel.pkl','rb')
p_model=open('model.pkl','rb')
p_model2=open('model2.pkl','rb')
p_model3=open('model3.pkl','rb')
p_model4=open('model4.pkl','rb')
stack_model=pickle.load(pickle_in)
model=pickle.load(p_model)
model2=pickle.load(p_model2)
model3=pickle.load(p_model3)
model4=pickle.load(p_model4)
# 2.Create the app object
app=FastAPI()

# 3.Index route, automatically open at http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message':'Bank Loan Approval'}


@app.post('/predict')
def predict_loan_status(data:BankLoan):
    data=data.dict()
    Gender=data['Gender']
    Married=data['Married']
    Dependents=data['Dependents']
    Education=data['Education']
    SelfEmployed=data['SelfEmployed']
    PropertyArea=data['PropertyArea']
    CreditHistory=data['CreditHistory']
    ApplicantIncome=data['ApplicantIncome']/100
    CoapplicantIncome=data['CoapplicantIncome']/100
    LoanAmount=data['LoanAmount']/10000
    Loan_Amount_Term=data['Loan_Amount_Term']
    
    
    
    p1=int(model.predict([[Gender,Married,Dependents,Education,SelfEmployed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,CreditHistory,PropertyArea]]))
    p2=int(model2.predict([[Gender,Married,Dependents,Education,SelfEmployed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,CreditHistory,PropertyArea]]))
    p3=int(model3.predict([[Gender,Married,Dependents,Education,SelfEmployed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,CreditHistory,PropertyArea]]))
    p4=int(model4.predict([[Gender,Married,Dependents,Education,SelfEmployed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,CreditHistory,PropertyArea]]))
    
    prediction=stack_model.predict([[p1,p2,p3,p4]])
    if(prediction==1):
        pred='You are Eligible of Loan'
    else:
        pred='You are not Eligible of Loan'
    return {'Loan Approval Prediction':pred}
    
# 4.Run API with uvicorn
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
    
# uvicorn main:app --reload
# uvicorn provides swaggerUI

