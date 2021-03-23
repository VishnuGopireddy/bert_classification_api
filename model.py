from transformers import BertModel, BertConfig
from torch import nn
import torch

class Net(nn.Module):
    def __init__(self, classes=2, pre_trained=False):
        super().__init__()
        self.config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.classes = classes
        self.linear = nn.Linear(768, classes)
        self.dropout = nn.Dropout(0.5)
        if pre_trained:
            print(f'Bert loading from {pre_trained}')
            self.bert_model = BertModel.from_pretrained(pre_trained, local_files_only=True)

    def forward(self, ids, attention_masks):
        x = self.bert_model(input_ids = ids, attention_mask = attention_masks)
        x = x['pooler_output']
        x = self.dropout(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    bert = Net()
    ids = torch.tensor([[101,102,156,186], [101,102,156,186]])
    masks = torch.tensor([[1,1,1,1],[1,1,1,1]])
    print(ids.shape)
    # print(bert(masks).shape)
    print(bert(ids, masks))