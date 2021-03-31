import tokenize 
import os
import re
import parser
from io import BytesIO
from sacrebleu import tokenize_v14_international
#special chars in python to help run tokenizer of strs
PYTHON_TOKEN2CHAR = {'STOKEN0': '#',
                     'STOKEN1': "\\n",
                     'STOKEN2': '"""',
                     'STOKEN3': "'''"
                     }

PYTHON_CHAR2TOKEN = {'#': ' STOKEN0 ',
                     "\\n": ' STOKEN1 ',
                     '"""': ' STOKEN2 ',
                     "'''": ' STOKEN3 '
                     }

#process strings
def str_proc(s,char2tok,tok2char,is_comm=False):
    s=s.replace(' ','_')
    for char,sp_tok in char2tok.items():
        s.replace(char,sp_tok)
    s = s.replace('\n', ' STRNEWLINE ')
    s = s.replace('\t', ' TABSYMBOL ')
    s = re.sub(' +', ' ', s)
    s = tokenize_v14_international(s)
    s = re.sub(' +', ' ', s)
    for special_token, char in tok2char.items():
        s = s.replace(special_token, char)
    s = s.replace('\r', '')

    return s

#tokens python code 
def py_tokenize(docs,keep_comments=False):
    for i in docs:
        tokens=[]
        itr=tokenize.tokenize(BytesIO(i.encode('utf-8')).readline)
        rem_docstr=0
        for toktype,tokval,_,_,line in itr:
            if toktype==tokenize.ENCODING or toktype==tokenize.NL :
                continue
            elif toktype==tokenize.COMMENT:
                continue

            elif toktype==tokenize.NEWLINE:
                if rem_docstr==1:
                    rem_docstr=0
                    continue
                tokens.append("NEW_LINE")
            elif toktype==tokenize.INDENT:
                tokens.append("INDENT")

            elif toktype==tokenize.DEDENT:
                #filter out empty blocks
                if tokens[-1]=='INDENT':
                    tokens[:-1]    
                else:
                    tokens.append('DEDENT') 

            elif toktype==tokenize.STRING:
                if tokval==line.strip():
                    #add comment handling later
                    rem_docstr=1
                else:
                    tokval=str_proc(tokval,PYTHON_CHAR2TOKEN,PYTHON_TOKEN2CHAR,False)
                    #print(tokval)
                    tokens.append(tokval)
                   
            
            elif toktype==tokenize.ENDMARKER:
                tokens.append("ENDMARKER")
                break
            else:
                tokens.append(tokval)
            
        assert(tokens[-1]=="ENDMARKER"),"ERROR,NO END MARRKER"
        print(len(tokens)-1," DONE BISHHHH")

        return tokens[:-1]

def extract_func_py(tokens):
    fn_start=0
    ind_cnt=0
    has_ind=0
    new_toks=[]
    for i in tokens:
        
        if ("def" in i):        
            fn_start=1
        if fn_start==1 :
            new_toks.append(i)
        if fn_start==1 and i=="INDENT":
            ind_cnt+=1
            has_ind=1
        if fn_start==1 and i=="DEDENT":
            ind_cnt-=1
        if ind_cnt==0 and has_ind==1:
            fn_start=0
            has_ind=0

    return new_toks    


if __name__ == "__main__":
    
    path="/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/python"
    f=os.listdir(path)
    print(f)
    
    docs=parser.parse_json(os.path.join(path,f[0]))
    tokens=py_tokenize(docs)
    tokens_fn_only=extract_func_py(tokens)
    print(len(tokens_fn_only))
