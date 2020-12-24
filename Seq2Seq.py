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
        
    def forward(self, src):
        
        #src = [src len, batch size] # batch size로 들어가지 않음 -> 수정
        
        embedded = self.dropout(self.embedding(src)) # error??
        
        #embedded = self.embedding(src)
        #embedded = embedded.to(self.deivce) # src len, 64,32
        outputs, (hidden, cell) = self.rnn(embedded) # error...?
        
        #hidden = hidden.to(self.deivce)
        #cell = cell.to(self.deivce)
        
        #outputs = [src len, batch size, hid dim * n directions] 
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell #.clone()
    
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout) #, batch_first=True
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        #self.init_weights()
    
    #def init_weights(self):
    #    torch.nn.init.xavier_uniform(self.embedding.weight)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0).type(torch.cuda.LongTensor) #.clone()
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) # error
        #embedded = embedded.clone()
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        '''
        output = output.clone()
        hidden = hidden.clone()
        cell = cell.clone()
        output[output!=output] = 0
        hidden[hidden!=hidden] = 0
        cell[cell!=cell] = 0
        '''
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0)) #.clone()
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell #.clone()
    
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
        
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[0] 
        trg_len = trg.shape[1] 
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        #outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) #to('cuda:0')
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device) 
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src) # # hidden, cell에서 nan 발생
    
        # fill nan to 0
        #hidden[hidden!=hidden] = 0
        #cell[cell!=cell] = 0
        
        #first input to the decoder is the <sos> tokens
        decoder_input = trg[0,:]
        
        #print(batch_size, trg_len, trg_vocab_size, hidden.shape, cell.shape, input.shape) # 61,100,100,2 100 512, 100
        
        for t in range(trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(decoder_input, hidden, cell) #[-1].unsqueeze(0) 
            #output = output.clone()
            
            #place predictions in a tensor holding predictions for each token
            
            # fill nan to 0
            #output[output!=output] = 0
            
            outputs[t] = output # nan 발생
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            #top1 = output.argmax(1) # 여기? 아래 참고
            chosen  = torch.argmax(output, dim=1)
            top1 = chosen
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            decoder_input = trg[t] if teacher_force else top1
            #input = input.clone()
        #outputs = outputs.clone()    
        #outputs[outputs!=outputs] = 0
        
        return outputs #torch.nan_to_num(outputs)