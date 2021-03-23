import pandas as pd
import numpy as np
import utils_preprocess as utils
from sklearn.model_selection import train_test_split

class PreProcess:
    def __init__(self):
        self.train_df = pd.read_csv('/home/vishnu/learning/sentiment140/training.1600000.processed.noemoticon.csv', header=None)
        self.test_df1 = pd.read_csv('/home/vishnu/learning/sentiment140/testdata.manual.2009.06.14.csv',header=None)

    def remove_id_from_tweet(self,tweet):
        tweet = utils.strip_smiles(tweet)
        tweet = utils.strip_links(tweet)
        tweet = utils.strip_all_entities(tweet)
        return tweet

    def prepare_dataset(self):
        self.train_df.drop([2,3,4,],axis=1,inplace=True)
        self.test_df1.drop([0,2,3,4],axis=1,inplace=True)
        self.train_df['pre_process'] = self.train_df[5].apply(self.remove_id_from_tweet)
        self.test_df1['pre_process'] = self.test_df1[5].apply(self.remove_id_from_tweet)
        train_df, val_df = train_test_split(self.train_df, test_size=0.4)
        val_df, test_df = train_test_split(val_df, test_size=0.5)
        self.train_df = train_df
        self.val_df = val_df
        self.test_df2 = test_df
        print('train_size', self.train_df.shape)
        print('val_size', self.val_df.shape)
        print('test_size 1', self.test_df1.shape)
        print('test_size 2', self.test_df2.shape)

if __name__ == '__main__':
    obj = PreProcess()
    obj.prepare_dataset()
    print(obj.test_df1.head())