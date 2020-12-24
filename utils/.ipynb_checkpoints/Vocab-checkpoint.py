import torch
from konlpy.tag import Mecab
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
import re

class Vocab():
    def __init__(self):

        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.tok=Mecab()
        _, self.vocab = get_pytorch_kogpt2_model()
    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_features(self, batch, sent_trunc=64, doc_trunc=100, split_token='\n'): # 100->64, sent trunc = batch size
        
        sents_list = []
        batch_sents = []
        # src
        for doc in batch['src']: # eunumerate 로 한번에 해결?
            #sents = doc.split(split_token) # line
            #sent=re.sub('','',doc)
            words = self.tok.morphs(doc) # change
            
            max_sent_len = min(doc_trunc, len(words))
            if len(words) > max_sent_len:
                words = words[:max_sent_len]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = self.vocab[sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)
            
        features = torch.LongTensor(features)
        
        sents_list = []
        max_sent_len = 0
        batch_sents2 = []
        # tgt
        for doc in batch['tgt']:
            #sents = doc.split(' ') # word
            #sent=re.sub('','',doc)
            words = self.tok.morphs(doc)
            
            #max_sent_num = min(doc_trunc, len(words))
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents2.append(words)

            #sents = sents[:max_sent_num]            
            #sents_list += sents
            #sents_list.append(sents)
        targets = []
        for sent in batch_sents2:
            target = self.vocab[sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            targets.append(target)
        
        targets = torch.LongTensor(targets)
        
        
        return features, targets #, doc_lens

    def make_predict_features(self, batch, sent_trunc=100, doc_trunc=200, split_token='\n'):

        sents_list, doc_lens = [], []
        for doc in batch['doc']:

            sents = doc.split(split_token)
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            sents_list += sents
            doc_lens.append(len(sents))

        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []

        for sent in sents_list:
            sent = re.sub('·', '', sent)
            if (len(sent) == 0):
                sent = "."

            words = self.tok.morphs(sent)
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature =self.vocab[sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)

        return features, doc_lens