import torch
import numpy as np
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

from Seq2Seq import *
from utils import Dataset,util,Vocab
import argparse
from torch.autograd import Variable

from tqdm import tqdm

import pandas as pd 

#device = "cuda:0"
#wantIllegalAccessException = True

train_path="./data/train.jsonl"
test_path="./data/abstractive_test_v2.jsonl"
out_path="./data/abstractive_sample_submission_v2.csv"

#import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.backends.cudnn.benchmark = False

SEED  = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def train(model, dataloader, optimizer, criterion, clip, vocab, device): # vocab 추가
    
    model.train()
    epoch_loss = 0
    
    t = tqdm(dataloader) # enumerate 안에 tqdm으로
    
    for i, batch in enumerate(t): 
        
        features, targets = vocab.make_features(batch, False) 
        features, targets = Variable(features.to(device)), Variable(targets.to(device))
            
        src = features 
        trg = targets 
        
        # src : batch, src len
        # trg : batch, trg len
        
        output = model(src, trg) 
        
        output_dim = output.shape[-1]
        
        output = output.view(-1, output_dim).clone() 
        trg = trg.view(-1).clone() 
        
        loss = criterion(output, trg) 
        
        t.set_description('(Loss: %g)' % loss)
        
        optimizer.zero_grad()
        loss.backward() 
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, vocab, device): # valid set
    
    model.eval()
    epoch_loss = 0
    
    t = tqdm(dataloader)
    
    with torch.no_grad():
        for i, batch in enumerate(t):   
            
            features, targets = vocab.make_features(batch, False) 
            
            features, targets = Variable(features.to(device)), Variable(targets.to(device))
            
            src = features.clone()
            trg = targets.clone()
            
            output = model(src, trg) 
            output_dim = output.shape[-1]
            
            #print('before : ', output.shape, trg.shape) # 64 64 80000 / 64 64
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)
            
            #print('after : ', output.shape, trg.shape) # 4096 80000 / 4096
            
            loss = criterion(output, trg)
            
            t.set_description('(Loss: %g)' % loss)
            
            epoch_loss += loss.item()
        
    return (epoch_loss / len(t)) 

def test(model, dataloader, criterion, vocab, device): # test set에 대한 생성 요약
    
    model.eval()
    
    t = tqdm(dataloader)
    abs_sent = np.empty(1, dtype=object)
    total_abs_sents = [] #np.empty(len(dataloader), dtype=object)
    
    with torch.no_grad():
        for i, batch in enumerate(t):   
            
            features, targets = vocab.make_features(batch, True) # batch size 1
            
            features, targets = Variable(features.to(device)), Variable(targets.to(device))

            src = features #.clone()
            trg = targets #.clone()
            
            output = model(src, trg) 
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)
            
            abs_sent = vocab.make_sents(output) # 요약된 문장
            total_abs_sents = np.append(total_abs_sents, abs_sent)
            
    return total_abs_sents  

def save_csv(out_path, abs_sents) : # test set 에서 요약된 문장 저장
    out = pd.read_csv(out_path)
    out['summary'] = np.array(abs_sents).reshape(-1) #[x for x in abs_sents] #abs_sents
    out.to_csv("./data/out_final.csv", mode='w', index=False)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
def main(N_EPOCHS,learning_rate,batch_size,device,save_dir) : 
    
    dataset = util.load_jsonl(train_path)
    dataset = Dataset.Dataset(dataset)
    
    train_length = int(len(dataset)*0.8)
    valid_length = len(dataset) - train_length
    train_set, val_set = torch.utils.data.random_split(dataset, (train_length, valid_length))
    # default batch size 보다 작은 부분 남았을 경우 check
    #train_set, val_set,_ = torch.utils.data.random_split(dataset, (32, 32,len(dataset)-64))

    test_set = util.load_jsonl(test_path)
    test_set = Dataset.Dataset(test_set)
    # 생성된 문장 차원 check
    #test_set,_ = torch.utils.data.random_split(test_set, (2,len(test_set)-2))
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size) # shuffle=True
    
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1) # 한줄씩 생성요약
    
    # input dim, output dim 변경
    INPUT_DIM = 49990 #80000
    OUTPUT_DIM = 49990 #80000
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    vocab = Vocab.Vocab() 

    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT, device)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    
    torch.autograd.set_detect_anomaly(True) 

    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index = 0).to(device) # padding 
    
    #N_EPOCHS = 1
    CLIP = 1

    best_valid_loss = 100 #float('inf')
    '''
    for epoch in range(N_EPOCHS):

        #start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP, vocab, device)
        valid_loss = evaluate(model, val_dataloader, criterion, vocab, device) # , valid_sents

        #end_time = time.time()
        #epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, f'{save_dir}/seq2seq.pt')

        #print(f'Epoch: {epoch+1:2} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    '''
    model = torch.load(f'{save_dir}/seq2seq.pt')
    test_sents = test(model, test_dataloader, criterion, vocab, device)
    
    save_csv(out_path, test_sents)
    
    
        
if __name__ == '__main__':
    import argparse
    #if not wantIllegalAccessException:
    #    torch.cuda.set_device(device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--N_EPOCHS', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001) 
    parser.add_argument('--batch_size', type=int, default=64) # 1024
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='save_dir')
    args = parser.parse_args()
    main(args.N_EPOCHS,
         args.learning_rate,
         args.batch_size,
         args.device,
         args.save_dir) 