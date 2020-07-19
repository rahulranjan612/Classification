
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

templates = Jinja2Templates(directory='template')
test_ventorized = open('model/test_df2.pkl', 'rb')
test_cv = joblib.load(test_ventorized)

fraud_model = open('model/fraudulent_model2.pkl', 'rb')
model_cv = joblib.load(fraud_model)
app = FastAPI()

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: float = 10.5

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse(name='index.html', context={
        'request':request
    })


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.get('/predict/')
def predict():
    prediction = model_cv.predict(test_cv)
    print(prediction)
    return 'Hello'
    # resTop10 = prediction.tolist()[0:10]

    # if (len(resTop10) > 0):
    #     return {'predicted values':resTop10}