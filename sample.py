from fastapi import FastAPI
from pydantic import BaseModel
from inference import Infer_bert

app = FastAPI()

class Item(BaseModel):
    sent: str
    label: int

@app.get('/')
def index():
    return(f'Sentence prediction with Bert fine-tuned on Sentiment 140K dataset')

infer_obj = Infer_bert('/home/vishnu/learning/bert_classification/checkpoints/bert_model_classification_epoch_7.pth')
db = {}

@app.get('/database')
async def database():
    return db

@app.get('/predict/{sentence}')
def predict(sentence):
    label = infer_obj.predict(sentence)
    print(label)
    if label == 1:
        label = 'positive'
    if label == 0:
        label = 'negative'
    names(sentence, label)
    return f'{sentence} with {label} polarity'


@app.put('/predict/{sentence}')
async def names(sentence:str,label:str, item:Item):
    db.update({sentence : label})
    return f'{sentence} with {label} polarity'