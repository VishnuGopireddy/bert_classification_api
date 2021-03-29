import torch
import transformers
from model import Net

class Infer_bert:
    def __init__(self, model_path = '/home/vishnu/learning/bert_classification/checkpoints/bert_model_classification_epoch_7.pth'):
        BERT_PRE_TRAINED = 'bert-base-uncased'
        self.model_path = model_path
        self.tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PRE_TRAINED)
        # self.net = Net(pre_trained=model_path)
        self.net = Net()
        self.net.load_state_dict(torch.load(model_path, map_location='cuda'))
        self.net.to('cpu')


    def predict(self, sent):
        self.tokens = self.tokenizer.encode_plus(sent, max_length=200, return_tensors='pt', padding='max_length').to('cpu')
        label = self.net(self.tokens['input_ids'], self.tokens['attention_mask'])
        return label.argmax().detach().cpu().numpy()



if __name__ == '__main__':
    obj = Infer_bert()
    sent = 'Resize the input token embeddings when new tokens are added to the vocabulary'
    label = obj.predict(sent)
    print(label)