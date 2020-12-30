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
                
        if('abstractive' in ab) : # label이 존재
            ex2 = {'src': '\n'.join(ab['article_original']),'tgt': ab['abstractive']} # 수정한 부분: join 명령어로 튜플을 하나의 문장으로 결합
        else : # test, label이 없음
            ex2 = {'src': '\n'.join(ab['article_original']), 'tgt' : 'nan'}
            
        return ex2

    def __len__(self):
        return len(self.examples)
