import torch
from konlpy.tag import Mecab
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
import re

from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

class Vocab():
    def __init__(self):

        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'
        self.tok=Mecab()
        _, self.vocab = get_pytorch_kogpt2_model()
        
        self.tok_path = get_tokenizer()
        self.tok2 = SentencepieceTokenizer(self.tok_path,  num_best=0, alpha=0)

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_features(self, batch, test, sent_trunc=64, doc_trunc=100, split_token='\n'): # 100->64, sent trunc = batch size
        
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
        for sent in batch_sents: # word to num
            feature = self.vocab[sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)
            
        features = torch.LongTensor(features)
        
        max_sent_len = 0
        batch_sents2 = []
        # tgt
        for doc in batch['tgt']:
            
            #print(doc)
            
            if test : 
                targets = torch.zeros((1,64), dtype=torch.long)
                #targets = torch.LongTensor(targets)
                
            else :
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
        
        
        return features, targets #, batch_sents

    
    def make_sents(self, output): # num to word
        
        batch_sent = ''
    
        for pred in output: # 80000 # 49990 까지는 가능 -> embedding dim 줄여야함
            
            #sent_list = self.vocab.to_tokens(torch.argsort(pred,descending=False).squeeze().cpu().numpy().tolist())[:64] # tgt len
            #sent = ''
            sent = self.vocab.to_tokens(torch.argsort(pred,descending=False).squeeze().cpu().numpy().tolist())[0]
            #print(sent)
            
            #for i in range(len(sent_list)) : # ▁현대
            #    
            #    if sent_list[i].startswith('<') :
            #        continue
            #    if sent_list[i].endswith('다') :
            #        sent += sent_list[i].replace('▁', ' ')
            #        sent += '.'
            #        continue
            #    sent += sent_list[i].replace('▁', ' ')
            if sent.startswith('<') :
                continue
            if sent.endswith('다') :
                batch_sent += sent.replace('▁', ' ')
                batch_sent += '.'
                continue
            batch_sent += sent.replace('▁', ' ')
            
            #batch_sents.append(sent)
            
        #print(batch_sent)
        
               
        return batch_sent