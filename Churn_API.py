import json

import joblib
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel


model = joblib.load('churn_model.pkl')

app = FastAPI()

"""
CreditScore          int64
Age                  int64
Tenure               int64
Balance            float64
NumOfProducts        int64
HasCrCard            int64
IsActiveMember       int64
EstimatedSalary    float64
Exited               int64
"""

class requestbody(BaseModel):
    Credit_score: int
    Age : int
    Tenure: int
    Balance : float
    Num_of_products : int
    has_card : int
    is_active_member : int
    est_salary : float

@app.post('/predict')
def predict_churn(data:requestbody):
    info = [[data.Credit_score,data.Age,data.Tenure,data.Balance,data.Num_of_products,data.has_card,data.is_active_member,data.est_salary]]
    out = model.predict(info)
    return json.dumps(str(out))

