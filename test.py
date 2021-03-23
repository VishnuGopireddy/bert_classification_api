from dataloader import SentimentDataset
from torch.utils.data import DataLoader
from pre_process import PreProcess

import torch
from pre_process import PreProcess
from dataloader import SentimentDataset
from torch.utils.data import DataLoader
from model import Net
from torch import nn
import transformers
# from torch.utils.tensorboard import SummaryWriter
import os
# import wandb

class Test:
    def __init__(self, chpoint):
        self.max_seq = 100
        self.classifier = Net(classes=2)
        self.classifier.load_state_dict(torch.load(chpoint))
        if torch.cuda.is_available():
            self.classifier.to('cuda')
        self.prepare_dataset()

    def prepare_dataset(self):
        dat_obj = PreProcess()
        dat_obj.prepare_dataset()
        test_df = dat_obj.test_df2
        test_dataset = SentimentDataset(test_df, max_length=100, mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=True)


    def test_epoch(self):
        self.classifier.eval()
        print('Test', end='\t')
        total_loss = 0
        total_correct = 0
        total = 0
        print(f'Total test samples {len(self.test_loader)}')
        for i, data in enumerate(self.test_loader):
            with torch.no_grad():
                text, x, label = data
                ids, atten_mask = x[0].squeeze_(1).to('cuda'), x[1].squeeze_(1).to('cuda')
                label = label.to('cuda')
                label_map = {0:'pos', 1:'neg'}
                polarity = label_map[label.to('cpu').numpy()[0]]
                print(f'{text[:100]} --> {polarity}')
                pred = self.classifier(ids, atten_mask)
                pred_classes = torch.argmax(pred, dim=1)
                correct = (label == pred_classes).sum()
                total_correct += correct
                total += label.shape[0]

        epoch_acc = (total_correct / total) * 100
        print(f'Accuracy on Test dataset is {epoch_acc}')

if __name__ == '__main__':
    ckpoint = 'checkpoints/bert_model_classification_epoch_6.pth'
    test_obj = Test(ckpoint)
    test_obj.test_epoch()