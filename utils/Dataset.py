import csv
import torch
import torch.utils.data as data
from torch.autograd import Variable
from .Vocab import Vocab
import numpy as np
from konlpy.tag import Mecab
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
import re


class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset, self).__init__()
        # data: {'sents':xxxx,'labels':'xxxx', 'summaries':[1,0]}
        self.examples = examples
        self.training = False
        
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.tok=Mecab()
        _, self.vocab = get_pytorch_kogpt2_model()

    def train(self):
        self.training = True
        return self

    def test(self):
        self.training = False
        return self

    def shuffle(self, words):
        np.random.shuffle(words)
        return ' '.join(words)

    def dropout(self, words, p=0.3):
        l = len(words)
        drop_index = np.random.choice(l, int(l * p))
        keep_words = [words[i] for i in range(l) if i not in drop_index]
        return ' '.join(keep_words)

    def __getitem__(self, idx):
        ab = self.examples[idx]
                
        if('abstractive' in ab) :
            #ex2 = ab[['article_original', 'abstractive']] # df
            #ex2 = dict(zip(ab.article_original, ab.abstractive))
            
            #ex2 = ab.to_dict('list')
            #ex2 = dict(zip(ab.article_original, ab.abstractive))
            
            ex2 = {'src': '\n'.join(ab['article_original']),'tgt': ab['abstractive']} # 수정한 부분: join 명령어로 튜플을 하나의 문장으로 결합
            
            #ex2 = ex2[['article_original', 'abstractive']]
            
            #ex2['src'] = self.token(ex2['src'])
            #ex2['tgt'] = self.token_abs(ex2['tgt'])
            
            #return ex2
        else : # test
            ex2 = {'src': '\n'.join(ab['article_original']), 'tgt' : 'nan'}
            
        return ex2

        #words = ex['sents'].split()
        #guess = np.random.random()

        #if self.training:
        #    if guess > 0.5:
        #        sents = self.dropout(words,p=0.3)
        #    else:
        #        sents = self.shuffle(words)
        #else:
        #    sents = ex['sents']
        #return {'id':ex['id'],'sents':sents,'labels':ex['labels']}
    '''
    def __getitem__(self, idx):
        ex = self.examples[idx]


        label = ''
        if('extractive' in ex):
            for i in range(len(ex['article_original'])):
                if i in ex['extractive']:
                    label += '1\n'
                else:
                    label += '0\n'
            ex2 = {'doc': '\n'.join(ex['article_original']) ,'labels': label.strip() , 'summaries': ex['abstractive']}

            return ex2

        else:

            ex2 = {'doc': '\n'.join(ex['article_original'])}
            return ex2
    '''
    def __len__(self):
        return len(self.examples)
