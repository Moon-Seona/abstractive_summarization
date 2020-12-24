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


# attention is all you need
def train(model, dataloader, optimizer, criterion, clip, vocab, device): # vocab 추가
    
    model.train()
    epoch_loss = 0
    
    t = tqdm(dataloader) # enumerate 안에 tqdm으로
    
    for i, batch in enumerate(t): # tqdm 은 enumerate 안에 넣는다!!
        
        #print(len(batch))
        
        features, targets = vocab.make_features(batch) #doc_lens        
        features, targets = Variable(features.to(device)), Variable(targets.to(device))
        
        src = features.clone()
        trg = targets.clone()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim).clone()
        trg = trg[1:].view(-1).clone()
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        #nan_mask = torch.isnan(output)
        #loss = criterion(output[~nan_mask], trg[~nan_mask[:,0]]) 
        
        loss = criterion(output, trg) 
        
        #if torch.isnan(output).sum() > 0 :
        #    print(output)
        #else :
        #    print(loss)
            #print(loss) # 4.3630, 4.2724
        
        t.set_description('(Loss: %g)' % loss)
        
        # loss, src, trg, output : cuda:0
        optimizer.zero_grad()
        loss.backward() # loss nan -> solved, copy_if failed to synchronize: cudaErrorIllegalAddress: an illegal memory access was encountered
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
        
def main(N_EPOCHS,learning_rate,batch_size,device,save_dir) : #
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = util.load_jsonl(train_path)
    #dataset = util.load_json_asDataFrame(train_path)
    dataset = Dataset.Dataset(dataset)
    
    #dataset = np.array(dataset) # 해야하나...?
    
    train_length = int(len(dataset)*0.8)
    valid_length = len(dataset) - train_length
    train_set, val_set = torch.utils.data.random_split(dataset, (train_length, valid_length))

    #train_set =  Dataset.Dataset(train_set)
    #val_set =  Dataset.Dataset(val_set)
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size) # shuffle=True
    
    # input dim, output dim 변경
    INPUT_DIM = 80000
    OUTPUT_DIM = 80000
    ENC_EMB_DIM = 32
    DEC_EMB_DIM = 32
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    vocab = Vocab.Vocab() # 여기로 이동

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

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP, vocab, device)
        valid_loss = evaluate(model, val_dataloader, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut6-model.pt')

        print(f'Epoch: {epoch+1:2} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
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
         args.save_dir) #