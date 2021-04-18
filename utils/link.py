import torch 
from parser import read_file


def create_dict(vocab):
    #create the vectors + link each
    dic={}
    val=0
    for i in vocab:
        word=i.split()[0]
        dic[word]=val
        val+=1
    return dic,len(vocab)

if __name__ == "__main__":
    vocab=read_file("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/py_train_vocab")
    dict_py,_=create_dict(vocab)
    print(dict_py)