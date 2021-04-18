import json
import os 
def get_keyword_list(path):
    l=[]
    with open(path) as d_file:
        for i in d_file:
            
            l.append(i[:-1].strip())   
    return l
def parse_json (path):
    
    
    docs=[]
    with open(path) as data_file:
        data=json.load(data_file)
        for i in data:
            docs.append(i["content"])

        #print(len(docs))    
    
    return docs
def read_file(path):
    
    with open(path) as data_file:
        data_l=list(data_file)
    return data_l
if __name__ == "__main__":
    
    path="/home/armaan/Desktop/DNoesktop_Files/transcoderplus/dataset/cpp"
    f=os.listdir(path)
    print(f)
    
    #docs=parse_json(os.path.join(path,f[0]))
    #print(docs[0])
    keywords=os.path.join(path,f[1])
    l=get_keyword_list(keywords)
    print(l)
