from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

class SentimentDataset(Dataset):
    def __init__(self, dataset, max_length=200, mode='train'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.df = dataset
        self.max_length = max_length
        self.mode = mode

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text, label = row['pre_process'], row[0]
        if label != 0:
            label = 1
        out_dict = self.tokenizer.encode_plus(text=text,
                                              padding='max_length',
                                              max_length=200,
                                              return_tensors='pt')
        # print(out_dict)
        if self.mode != 'test':
            return [(out_dict['input_ids'][:, :self.max_length], out_dict['attention_mask'][:, :self.max_length]), label]
        else:
            return [text, (out_dict['input_ids'][:, :self.max_length], out_dict['attention_mask'][:, :self.max_length]), label]

    def __len__(self):
        # return int(self.df.shape[0])
        return 2000

if __name__ == '__main__':
    from pre_process import PreProcess
    dat_obj = PreProcess()
    dat_obj.prepare_dataset()
    train_df = dat_obj.train_df
    dataset = SentimentDataset(train_df, 200)
    train_loader = DataLoader(dataset, batch_size=5,num_workers=8)

    for i, j in enumerate(train_loader,0):
        print(i)
        print(j[0][0])