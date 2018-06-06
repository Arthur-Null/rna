import os
import re
import pickle

path = 'RNA_trainset/'
def readfile(file):
    f = open(file,"r")
    file_list = f.readlines()
    # print(file_list)
    return file_list

def writefile(file,context):
    f = open(file,"w")
    f.write(context)
#

dic = {}
tmp = 0
for file in os.listdir(path):
    # path2 = str(path + file + "/train")
    # if os.path.exists(path2):
    #     continue
    rna_list = readfile(path + file + "/train")
    print(len(rna_list),file)

    rna_input = []
    rna_label = []
    rna_output = []
    for i in rna_list:
        rna = re.split(r'[\s]', i)
        rna_input.append(rna[0])
        if len(rna) == 1:
            rna_label.append(' ')
        else:
            rna_label.append(rna[1])
    print(len(rna_label), len(rna_input))

    for i in range(len(rna_input)):
        if rna_input[i] in dic.keys():
            rna_output.append(str(dic[rna_input[i]]+'\t' + str(rna_label[i])))
        else:
            writefile("test", rna_input[i])
            r = os.popen(str("RNAfold --noPS  test"))
            info = r.readlines()
            rna_output.append(str(info[1]) + '\t' + str(rna_label[i]))
            dic[rna_input[i]] = info[1]
            tmp += 1
            print(tmp)
        # print(i)
    str_list = [line + '\n' for line in rna_output]
    f = open(str(path + file + "/train_2nd"), 'w')
    f.writelines(str_list)