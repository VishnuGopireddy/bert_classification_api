from fastapi import FastAPI, Request, Form
from pydantic import BaseModel
from inference import Infer_bert
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates/")

app = FastAPI()
class Item(BaseModel):
    sent: str
    label: int

# @app.get('/')
# def index():
#     welocome = f'Sentence prediction with Bert fine-tuned on Sentiment 140K dataset'
#     return '<html> <p>Sentence prediction with Bert fine-tuned on Sentiment 140K dataset.</p> </html>'

infer_obj = Infer_bert('/home/vishnu/learning/bert_classification/checkpoints/bert_model_classification_epoch_7.pth')
db = {}

@app.get("/")
def form_post(request: Request):
    result = "Enter a sentence to predict"
    return templates.TemplateResponse('forms.html', context={'request': request, 'result': result})

@app.post("/")
def form_post(request: Request, sentence: str = Form(...)):
    label = infer_obj.predict(sentence)
    print(label)
    if label == 1:
        label = 'positive'
    if label == 0:
        label = 'negative'
    # names(sentence, label)
    result = label
    return templates.TemplateResponse('forms.html', context={'request': request, 'result': result, 'sentence': sentence})

# @app.get('/database')
# async def database():
#     return db
#
# @app.get('/predict/{sentence}')
# def predict(sentence):
#     label = infer_obj.predict(sentence)
#     print(label)
#     if label == 1:
#         label = 'positive'
#     if label == 0:
#         label = 'negative'
#     names(sentence, label)
#     return f'{sentence} with {label} polarity'
#
#
# @app.put('/predict/{sentence}')
# async def names(sentence:str,label:str, item:Item):
#     db.update({sentence : label})
#     return f'{sentence} with {label} polarity'