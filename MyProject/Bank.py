# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:08:14 2022

@author: Engineer Sameer
"""

from pydantic import BaseModel
# Class which provides all input features to my ML Model

class BankLoan(BaseModel):
    Gender: int # 0-Female 1-Male
    Married: int # 0-No 1-Yes
    Dependents : int # 0,1,2,4
    Education : int # Graduate-0 Not-Graduate-1
    SelfEmployed : int # Self-1 Not-0
    PropertyArea : int # SemiUrban-2 Urban-1 Rural-0
    CreditHistory: int # 1-Yes 0-No
    ApplicantIncome : float 
    CoapplicantIncome : float
    LoanAmount : float 
    Loan_Amount_Term : float
    
    
    
