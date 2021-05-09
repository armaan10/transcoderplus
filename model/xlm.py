#embd_dim=1024 o/p!!
#15 %
#each batch is 32 seqs of 512 tokens
#pad token mask token
import sys
sys.path.insert(1, '/home/armaan/Desktop/Desktop_Files/transcoderplus/utils')
import torch
import torch.nn as nn
import link
import parser
import random
import time
import math
class EncoderNet(nn.Module):
    def __init__ (self,vocab_size,model_dim,att_h,output_dim,num_layers=1):
        super(EncoderNet,self,).__init__()
        self.encoder_layer=nn.TransformerEncoderLayer(model_dim,att_h,output_dim)   
        self.encoder=nn.TransformerEncoder(self.encoder_layer,num_layers)
        self.pos_enc=PositionalEncoding(model_dim)
        self.emd_py=nn.Embedding(vocab_size[0],1024)
        self.emd_cpp=nn.Embedding(vocab_size[1],1024)
        self.linear_py=nn.Linear(1024,vocab_size[0])
        self.linear_cpp=nn.Linear(1024,vocab_size[1])
    #pad_idx 1 -> pad it there mask_idx 1-> masked idx    
    def forward(self,token_idx,mask_matrix,pad_idx,lang):
        if lang=="py":
            inp_vec=self.emd_py(token_idx)
        else :
            inp_vec=self.emd_cpp(token_idx)
        assert inp_vec.size()==mask_matrix.size()
        #create random vector for mask token
        inp_vec=(inp_vec*mask_matrix)
        inp_vec=self.pos_enc(inp_vec)
        output=self.encoder(inp_vec)
        if lang=="py":
            output=self.linear_py(output)
        else:
            output=self.linear_cpp(output)
        output=nn.Softmax(output)
        return output

        



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def make_batches(tokens,no_seqs=32,inp_l=16):
    no_iters=int(len(tokens)/(no_seqs*inp_l))
    batches=[]
    total_batch_l=(no_seqs*inp_l)
    
    for i in range(no_iters):
        batch=tokens[i*total_batch_l:(i+1)*(total_batch_l)]
        seqs=[]
        for j in range(no_seqs):
            seqs.append(batch[j*inp_l:inp_l*(j+1)])

        batches.append(seqs)
    rem_batch=tokens[total_batch_l*no_iters:]
    no_iters=int(len(rem_batch)/inp_l)
    seqs=[]
    for i in range(no_iters):
        seqs.append(rem_batch[i*inp_l:(i+1)*inp_l])
    #pad rem_batch
    padded_seq= rem_batch[no_iters*inp_l:]+["[PAD]"]*abs(len(rem_batch[no_iters*inp_l:]) -inp_l)   
    seqs.append(padded_seq)
    batches.append(seqs)
    return batches

def mask_tokens(tokens,masking_per=15):
    no_mask_tokens=int(len(tokens)*masking_per/100)
    mask_idx=random.sample(range(0,len(tokens)),no_mask_tokens)
    masked_token_l=[]
    for i in mask_idx:
        masked_token_l.append(tokens[i])
        tokens[i]="[MASK]"
    return tokens,masked_token_l    
def make_targets(token_batch,dicto,vocab_vecs):
    batch_v=[]
    for batch in token_batch:
        seq_v=[]
        for seq in batch:
            
            
            seq_v.append(link.lookup(seq,dicto,vocab_vecs))

            #print(vecs.size())
        #print(torch.stack(seq_v))
        batch_v.append(torch.stack(seq_v))
    return batch_v
def get_tokens_idx(token_batch,dicto):
    batch_v=[]
    for batch in token_batch:
        seq_v=[]
        for seq in batch:
            
             l=torch.LongTensor([dicto[x] for x in seq ])
             seq_v.append(l)
        batch_v.append(torch.stack(seq_v))
    return batch_v
#in the form (batch,seqs,tokens)
def get_mask_pad_matrix(embd_dim,token_batch,no_seqs,seq_l):
    batch_mask=[]
    batch_pad=[]
    for batch in token_batch:
        mask_seq=[]
        zeros=torch.zeros(no_seqs,seq_l)
        for seq in batch:
            ones_matrix=torch.ones(len(seq),embd_dim)
            if "[MASK]" in seq:
                random_mask=torch.rand(1,embd_dim)
                ones_matrix[seq.index("[MASK]")]=random_mask
            mask_seq.append(ones_matrix)
        batch_mask.append(torch.stack(mask_seq))
        batch_pad.append(zeros)
    if "[PAD]" in token_batch[-1][-1]:
        seq=token_batch[-1][-1]
        pad_l=len(seq)-seq.index("[PAD]")
        pad_idx=torch.ones(1,pad_l)
        batch_pad[-1][-1,seq.index("[PAD]"):]=pad_idx
    return batch_mask,batch_pad

#encoder params def






#reading tokens and vocab
vocab_py=parser.read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/py_train_vocab")
tokens_py=parser.read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/py_train_bpe")
vocab_cpp=parser.read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/cpp_train_vocab")
tokens_cpp=parser.read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/cpp_train_bpe")

tokens_list_py=[]
tokens_list_cpp=[]
#imp part add later 
for i in tokens_py:
        tokens_list_py+=i.split()
for i in tokens_cpp:
        tokens_list_cpp+=i.split()
dict_py,size_py=link.create_dict(vocab_py)
dict_cpp,size_cpp=link.create_dict(vocab_cpp)
special_tks=["[MASK]","[PAD]"]
dict_py[special_tks[0]]=size_py
dict_py[special_tks[1]]=size_py+1
size_py+=2
dict_cpp[special_tks[0]]=size_cpp
dict_cpp[special_tks[1]]=size_cpp+1
size_cpp+=2


"""

one_hot_vecs_cpp=torch.eye(size_cpp)
s=mask_tokens(tokens_list_py)
s=make_batches(tokens_list_py)
m,p=get_mask_pad_matrix(1024, s, 32, 16)
t=make_targets(s, dict_py, one_hot_vecs_py)
#d=get_tokens_idx(s, dict_py)
print(p[-1][-1],s[-1][-1])
"""
#encoder params def
vocab_size=[size_py,size_cpp]
model_dim=1024
att_h=8
output_dim=1024
num_layers=6
lr=0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=EncoderNet(vocab_size, model_dim, att_h, output_dim).to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr)

#training params
no_seqs=32
seq_l=16
"""
embedding_py=nn.Embedding(size,1024)
b=make_batches(tokens_list_py,31,16)
print(len(b[0][-2]))
s=torch.LongTensor([[1,2,3],[1,2,5]])
print(size,embedding_py(torch.LongTensor(s)).size())
"""
#each iter mask then batchify 





def train(epochs=10):
    model.train()
    start_time=time.time()
    unmasked_train_set_py=make_batches(tokens_list_py)
    unmasked_train_set_cpp=make_batches(tokens_list_cpp)
    targets_py=make_targets(unmasked_train_set_py,dict_py,one_hot_vecs_py)
    targets_cpp=make_targets(unmasked_train_set_cpp,dict_cpp,one_hot_vecs_cpp)
    for ep in range(epochs):
        model.train()
        #mask tokens
        masked_tokens_py=mask_tokens(tokens_list_py)
        masked_tokens_cpp=mask_tokens(tokens_list_cpp)
        #make into batches
        masked_batch_py=make_batches(mask_tokens_py)
        masked_batch_cpp=make_batches(mask_tokens_cpp)
        #get padding and masking matrix
        mask_py,pad_py=get_mask_pad_matrix(embd_dim, masked_batch_py, no_seqs, seq_l)
        mask_cpp,pad_cpp=get_mask_pad_matrix(embd_dim, masked_batch_cpp, no_seqs, seq_l)
        #get token indexes
        masked_idx_py=get_tokens_idx(masked_batch_py, dict_py)
        masked_idx_cpp=get_tokens_idx(masked_batch_cpp, dict_cpp)        
        #alternate batches 
        loss_total=0          
        for iter_n in (len(masked_batch_py)+len(masked_batch_cpp)):
            py_iter=0
            cpp_iter=0
            if iter_n%2==0 or cpp_iter>=len(masked_batch_cpp) and py_iter<len(masked_batch):
                tokens_idx=masked_idx_py[py_iter]
                mask=mask_py[py_iter]
                pad=pad_py[py_iter]
                
                targets=targets_py[py_iter]
                py_iter+=1
                lang="py"
            else:
                tokens_idx=masked_idx_cppp[cpp_iter]
                mask=mask_cpp[cpp_iter]
                pad=pad_cpp[cpp_iter]
                targets=targets_cpp[cpp_iter]
                cpp_iter+=1
                
                lang="cpp"

            optimizer.zero_grad()
            output = model(tokens_idx,mask,pad,lang)
            loss=criterion(output,targets)
            loss_total+=loss
            loss.backward()
            optimizer.step()
        if e%5 ==0:
            print("loss:",loss/5)


train()