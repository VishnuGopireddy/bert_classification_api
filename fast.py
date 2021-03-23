import uvicorn
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
from enum import Enum

app = FastAPI()

class Models(str, Enum):
    Bert = 'Bert'
    GPT = 'GPT2'
    seq_seq = 'seq-seq'

class Item(BaseModel):
    name: str
    age: int
    degree: bool

db = [{'name': 'vishnu',
       'age': 24,
       'degree': True},

      {'name':'RAvi',
       'Age':30,
       'degree':False}
      ]

@app.get('/model/{model_name}')
def models(model_name:Models):
    if model_name == Models.GPT:
        return 'GPT model'


@app.get('/')
def welcome():
    return 'Welcome'

@app.get('/item')
def show_db():
    print('db',db)
    return db

@app.get('/item/{item_id}')
def print_db(db_id:int):
    item = db[db_id]
    print('item',item)
    return item

@app.put('/items/add')
def add_item(item:Item):
    db.append(item)
    return 'added_successfully'

@app.post('/items/')
def add_item_post(item:Item):
    db.append(item)
    return 'added_successfully post'
