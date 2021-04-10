import tokenize 
import os
import re
import parser
import random
from io import BytesIO
from sacrebleu import tokenize_v14_international
import clang
from clang.cindex import TokenKind
import fastBPE
clang.cindex.Config.set_library_path('/usr/lib/llvm-10/lib')
idx=clang.cindex.Index.create()
#special chars in python/java/cpp to help run tokenizer of strs
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

JAVA_TOKEN2CHAR = {'STOKEN0': "//",
                   'STOKEN1': "/*",
                   'STOKEN2': "*/",
                   'STOKEN3': "/**",
                   'STOKEN4': "**/",
                   'STOKEN5': '"""',
                   'STOKEN6': '\\n'
                   }
JAVA_CHAR2TOKEN = {"//": ' STOKEN0 ',
                   "/*": ' STOKEN1 ',
                   "*/": ' STOKEN2 ',
                   "/**": ' STOKEN3 ',
                   "**/": ' STOKEN4 ',
                   '"""': ' STOKEN5 ',
                   '\\n': ' STOKEN6 '
                   }

CPP_TOKEN2CHAR = JAVA_TOKEN2CHAR.copy()
CPP_CHAR2TOKEN = JAVA_CHAR2TOKEN.copy()


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

#cpp handling 
def get_cpp_tokens(s):
    tokens = []
    
    s = s.replace(r'\r', '')
    hash = str(random.getrandbits(128))
    parsed_code = idx.parse(hash + '_tmp.cpp', args=['-std=c++11'], unsaved_files=[(hash + '_tmp.cpp', s)], options=0)
    for tok in parsed_code.get_tokens(extent=parsed_code.cursor.extent):
        tokens.append((tok.spelling, tok.kind))
    return tokens

def cpp_tokenize(docs,keep_comments=False):
    tokens=[]
    j=0
    for i in docs:
        assert isinstance(i,str)
        toks_coll=get_cpp_tokens(i)
       
    
        for tok,typ in toks_coll:
            if typ==TokenKind.COMMENT:
                continue
            if typ == TokenKind.LITERAL:
                tok=str_proc(tok,CPP_CHAR2TOKEN,CPP_TOKEN2CHAR)
                tokens.append(tok)
                
            else:
                tokens.append(tok)
        print(j)
        j+=1
    return tokens

def collect_fn_cpp(doc_lines,start_pos):
    braces_cnt=0
    has_braces=0
    docs="\n"
     
    for i in range(start_pos,len(doc_lines)):
        
        
        
        
        if "{" in doc_lines[i] :
            braces_cnt+=1
            has_braces=1
        elif "}" in doc_lines[i]:
            braces_cnt-=1
        if braces_cnt==0 and has_braces ==1 :
            break
    return docs.join(doc_lines[start_pos:i+1])       
        
def extract_func_cpp(docs,keywords_path):
    docs_fn_only=[]

    for i in docs:
        docs_lines = i.split("\n")
        #add -1 later to range
        
        
        for j in range(len(docs_lines)-1):
            
            
            if re.search(r"\(([^)]+)\)",docs_lines[j]) and "{" in docs_lines[j+1]:
                temp_doc_split=docs_lines[j].split()
                pos = 0
                flag=0
                
                for k in temp_doc_split:
                    if "(" in k:
                        if k[0]=="(":
                            pos-=1
                            flag=1
                        break
                    pos+=1
                
                if flag==0:
                    key_w=temp_doc_split[pos][:temp_doc_split[pos].find("(")]
                    #print(key_w)
                else :
                    key_w=temp_doc_split[pos]
                if key_w in keywords or ";" in temp_doc_split[-1]:
                    #print(k)
                    continue
                
                #print(docs_lines[j]) 
                
                fn_only=collect_fn_cpp(docs_lines,j)
                docs_fn_only.append(fn_only)
                
                #comment out
                
    return docs_fn_only
def test_regex(docs):
    #docs_fn_only=[]
    
    for i in docs:
        docs_lines = i.split("\n")
        
        for j in range(len(docs_lines)-1):
            #print(docs_lines[j])
            #print(j)
            if re.search(r"\(([^)]+)\)",docs_lines[j]) and "for" in docs_lines[j]:

                temp_fil=docs_lines[j].split()
                pos=0
                for k in temp_fil:
                    if "(" in temp_fil:
                        k-=1
                        break
                    k+=1
                #if doc
                return
            if j==50:
                return
            
if __name__ == "__main__":
    
    path="/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/cpp"
    f=os.listdir(path)
    print(f)
    
    docs=parser.parse_json(os.path.join(path,f[0]))
    keywords_path=os.path.join(path,f[1])
    keywords=parser.get_keyword_list(keywords_path)
   
    #test_regex(docs)
    #print(keywords)
    docs=extract_func_cpp(docs,keywords)
    
    
    tokens=cpp_tokenize(docs)
    
    f=open("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/fn_cpp_tokens.txt","w")
    for i in tokens:
        f.write("%s\n" % i)
    f.close()
    
    
    
    
       
        
    '''
    tokens_fn_only=extract_func_py(tokens)
    print(len(tokens_fn_only))
    f=open("/home/armaan/Desktop/Desktop_Files/transcoderplus/dataset/bpe_files/tokeinzed_cpp.txt","w")
    for i in tokens_fn_only:
        f.write("%s\n" % i)
    f.close()
    '''