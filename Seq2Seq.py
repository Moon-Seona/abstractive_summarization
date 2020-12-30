import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

SEED  = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = True

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.deivce = device
        self.emb_dim = emb_dim
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout, batch_first=True) # change
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform(self.embedding.weight)
        
    def forward(self, src): # batch, src len
        
        embedded = self.dropout(self.embedding(src)) # batch, src len, embed dim
        outputs, (hidden, cell) = self.rnn(embedded) # batch, src len, hid dim / layers, batch, hidden dim
        
        return hidden, cell 
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, batch_first=True) # , batch_first=True 
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, input, hidden, cell):
        
        input = input.unsqueeze(-1).type(torch.cuda.LongTensor) # batch, 1
        
        embedded = self.dropout(self.embedding(input)) # batch, 1, emb dim
        
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell)) 
    
        prediction = self.fc_out(output.squeeze())
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5): # batch, src len / batch, trg len
        
        batch_size = trg.shape[0] # 32
        trg_len = trg.shape[1] # 64
        trg_vocab_size = self.decoder.output_dim # 49990
        
        # batch, trg len, trg vocab size
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device) 
        
        hidden, cell = self.encoder(src) # 32, 100
    
        # batch size와 관련된 error의 시작....
        # 현재 모델에서 trg len, batch size를 동일하게 설정..
        decoder_input = trg[:,0] #[0,:] 
        
        # for문 전에 decoder input에 대한 설정만 바꾸고, 
        # for문 내의 decoder input, output에 대한 설정은 안바꿔서 error...
        for t in range(trg_len): # 타겟 길이만큼
            
            output, hidden, cell = self.decoder(decoder_input, hidden, cell) 
            
            if batch_size == 1 : #  test에서 batch를 1로 설정하여, decoder에서 squeeze를 할때 첫 차원의 1이 사라짐
                output = output.unsqueeze(0)
            
            # outputs[t] 에서 변경
            outputs[:,t,:] = output # output : batch, dec out dim
            
            teacher_force = random.random() < teacher_forcing_ratio
            
            chosen  = torch.argmax(output, dim=1)
            top1 = chosen
            
            # trg[t] 에서 변경
            decoder_input = trg[:,t] if teacher_force else top1 
            
        return outputs 