import re
import os
import gensim
import pickle
path = "RNA_trainset/"
def readfile(file):
    f = open(file,"r")
    file_list = f.readlines()
    return file_list

def k_mer(rna, length):
    seg_list = re.findall('.{' + str(length) + '}', rna)
    seg_list.append(rna[(len(seg_list) * length):])
    seg_list.remove('')
    return seg_list

def get_all_rna():
    dic = {}
    for file in os.listdir(path):
        rna_list = readfile(path + file + "/train")
        for i in rna_list:
            rna = re.split(r'[\s]', i)
            if dic.get(rna[0]):
                continue
            else:
                dic[rna[0]] = rna[0]
    return dic

def train_model(dic):
    sentence = []
    for i in dic:
        sentence.append(k_mer(dic[i],4))
    print(len(sentence))
    model = gensim.models.Word2Vec(sentence, min_count=1,size=10)
    model.save('mymodel')

# train_model(get_all_rna())
# model = gensim.models.Word2Vec(read_all_rna(), min_count=1)
# model.save('mymodel')

model = gensim.models.Word2Vec.load('mymodel')


for file in os.listdir(path):
    print(file)
    rna_list = readfile(path + file + "/train")
    rna_output = []
    num = 0
    for i in rna_list:
        num += 1
        if num%100 == 0:
            print(num)
        rna = re.split(r'[\s]', i)
        word = k_mer(rna[0],4)
        tmp =[]
        for words in word:
            tmp.append(model[words].tolist())
        rna_output.append(str(tmp) + '\t' + str(rna[1]))
    str_list = [line + '\n' for line in rna_output]
    output = open(str(path + file + "/train_2nd.pk1"), 'wb')
    pickle.dump(str_list,output)
