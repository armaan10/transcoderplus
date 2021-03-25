import json
import os 

def parse_json (path):
    
    
    docs=[]
    with open(path) as data_file:
        data=json.load(data_file)
        for i in data:
            docs.append(i["content"])

        #print(len(docs))    
    
    return docs

if __name__ == "__main__":
    
    path="/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/python"
    f=os.listdir(path)
    print(f)
    
    docs=parse_json(os.path.join(path,f[0]))

