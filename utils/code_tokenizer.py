import tokenize 
import os
import parser
from io import BytesIO
#tokens python code 
def py_tokenize(docs):
    for i in docs:
        
        tokens=tokenize.tokenize(BytesIO(i.encode('utf-8')).readline)
        for tokn,tokval,_,_,line in tokens:
            print(line)

        break
if __name__ == "__main__":
    
    path="/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/python"
    f=os.listdir(path)
    print(f)
    
    docs=parser.parse_json(os.path.join(path,f[0]))
    py_tokenize(docs)
