import torch 
from parser import read_file



''''size of vector total (vocab,embd_dim)'''


def create_dict(vocab):
    #create the vectors + link each
    dic={}
    val=0
    for i in vocab:
        word=i.split()[0]
        dic[word]=val
        val+=1
    return dic,len(vocab)
def lookup(tokens,dicto,vectors):
    idx_no=[dicto[x] for x in tokens]
    return vectors[idx_no]

if __name__ == "__main__":
    vocab=read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/py_train_vocab")
    dict_py,size=create_dict(vocab)
    tokens=read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/py_train_bpe")
    tokens_list=[]
    for i in tokens:
        
        tokens_list+=i.split()
    vectors=torch.rand(size,1)
    print(vectors[18])
    print(lookup(tokens_list[:2],dict_py,vectors))
    
    '''print(size)
    
    print(vectors[[12,]])
    x=["'",","]
    print(lookup(tokens[:2],dict_py,vectors))'''
    