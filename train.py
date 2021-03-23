import torch
from pre_process import PreProcess
from dataloader import SentimentDataset
from torch.utils.data import DataLoader
from model import Net
from torch import nn
import pkbar
import transformers
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import os
# import wandb

class TrainModel:
    def __init__(self):
        self.max_seq = 100
        self.classifier = Net(classes=2)
        if torch.cuda.is_available():
            self.classifier.to('cuda')
        self.criterion = nn.CrossEntropyLoss()
        self.optim = transformers.AdamW(self.classifier.parameters(), lr=5e-5)
        self.writer = SummaryWriter('logs/')
        self.val_epoch_step = 0
        self.epochs = 10
        self.log_file = 'logs/logs.txt'
        self.prepare_dataset()
        self.start_epoch = 0
        # wandb.login()
        # wandb.init(project="bert_sentiment")

    def resume_training(self):
        # try:
        if os.path.isfile(self.log_file):
            self.logwriter = open(self.log_file, 'r')
            ckpoints = self.logwriter.readlines()
            ckpoint = ckpoints[-1]
            if os.path.isfile(ckpoint):
                self.start_epoch = int(ckpoint.split('_')[-1].split('.')[0])+1
                print(f'checkpoints found resume from{ckpoint}')
                self.classifier.load_state_dict(torch.load(ckpoint, map_location='cuda'))
            else:
                print('No logs found start from epoch 0')
                self.start_epoch = 0
        else:
            print('No logs found start from epoch 0')
            self.start_epoch = 0

    def prepare_dataset(self):
        dat_obj = PreProcess()
        dat_obj.prepare_dataset()
        train_df = dat_obj.train_df
        val_df = dat_obj.val_df
        test_df = dat_obj.test_df1
        train_dataset = SentimentDataset(train_df, max_length=100)
        val_dataset = SentimentDataset(val_df, max_length=100)
        test_dataset = SentimentDataset(test_df, max_length=100, mode='test')
        self.train_loader = DataLoader(train_dataset, batch_size=32, num_workers=0, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0, shuffle=True)

    def val_epoch(self):
        self.classifier.eval()
        val_per_epoch = len(self.val_loader)
        print('Val', end='\t')
        val_pbar = pkbar.Kbar(target=val_per_epoch, epoch=self.start_epoch, num_epochs=self.epochs)
        total_loss = 0
        total_correct = 0
        total = 0
        for i, data in enumerate(self.val_loader):
            with torch.no_grad():
                x, label = data
                ids, atten_mask = x[0].squeeze_(1).to('cuda'), x[1].squeeze_(1).to('cuda')
                label = label.to('cuda')
                pred = self.classifier(ids, atten_mask)
                pred_classes = torch.argmax(pred,dim=1)
                loss = self.criterion(pred, label)
                correct = (label == pred_classes).sum()
                total_loss += loss.item()
                total_correct += correct
                total += label.shape[0]
                val_pbar.update(i, values=[('step loss', loss), ('step accuray', correct/label.shape[0])])

        epoch_loss = total_loss / (len(self.val_loader))
        epoch_acc = (total_correct/total)*100
        val_pbar.add(1, values=[('epoch loss', epoch_loss), ('epoch accuray', epoch_acc)])
        self.writer.add_scalar('val/loss', epoch_loss, global_step=self.val_epoch_step)
        self.writer.add_scalar('val/accuracy', epoch_acc, global_step=self.val_epoch_step)
        self.val_epoch_step += 1

    def train(self):
        # wandb.watch(self.classifier)
        self.resume_training()
        train_per_epoch = len(self.train_loader)
        for epoch in range(self.start_epoch, self.epochs):
            print('Train',end='\t')
            torch.save(self.classifier, 'checkpoints/bert_model_classification_epoch_{}.pth'.format(epoch))
            pbar = pkbar.Kbar(target=train_per_epoch, epoch=epoch, num_epochs=self.epochs)
            total_loss = 0
            total_correct = 0
            total = 0
            print('Train',end='\t')
            for i, data in enumerate(self.train_loader):
                x, label = data
                ids, atten_mask = x[0].squeeze_(1).to('cuda'), x[1].squeeze_(1).to('cuda')
                label = label.to('cuda')
                self.optim.zero_grad()
                pred = self.classifier(ids, atten_mask)
                pred_classes = torch.argmax(pred,dim=1)
                loss = self.criterion(pred, label)
                loss.backward()
                self.optim.step()
                correct = (label == pred_classes).sum()
                total_loss += loss.item()
                total_correct += correct
                total += label.shape[0]

                # if i % 2000 == 0:
                #     print('Info :')
                #     print(f'Predicted {pred_classes}',end=' ')
                #     print(f'correct {label}', end=' ')
                #     print(f'Accuracy {correct/label.shape[0]}', end=' ')
                #     print(f'Loss {loss}', end='\n')

                pbar.update(i, values=[('step loss', loss), ('step accuray', correct/label.shape[0])])
            # print('{}|{}: loss: {}, accuracy: {}'.format(epoch, epochs, total_loss, total_acc))
            epoch_loss = total_loss / len(self.train_loader)
            epoch_acc = (total_correct/total)*100
            pbar.add(1, values=[('epoch loss', epoch_loss), ('epoch accuray', epoch_acc)])
            self.writer.add_scalar('train/loss', epoch_loss, global_step=epoch)
            self.writer.add_scalar('train/accuracy', epoch_acc, global_step=epoch)
            ckpoint = 'checkpoints/bert_model_classification_epoch_{}.pth'.format(epoch)
            torch.save(self.classifier.state_dict(), ckpoint)
            print(f'checkpoints saved at {ckpoint}')
            self.logwriter = open(self.log_file, 'a')
            self.logwriter.writelines(f'\n{ckpoint}')
            self.logwriter.flush()
            # self.classifier.save('bert_model_classification_epoch_{}.pth'.format(epoch))
            self.val_epoch()
            self.start_epoch += 1

if __name__ == '__main__':
    train_obj = TrainModel()
    train_obj.train()
    # train_obj.resume_training()