from wordseg.wordseg import *

protein_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
                'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                'LIN28B',
                'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
                'WTAP', 'ZC3H7B']
test_list = {'AGO1'}
number_of_protein = len(protein_list)

def segment_data(data_path="../dataset/trainset/"):
    for iter_protein in range(number_of_protein):
        fin = open(data_path+protein_list[iter_protein],'r')
        fout = open(data_path+protein_list[iter_protein]+'_wordseg','w')
        fwords = open(data_path+protein_list[iter_protein]+'_words','w')
        #flog = open(data_path+protein_list[iter_protein]+'/ws_log','w')
        print('----'+protein_list[iter_protein]+'----')
        doc = ''
        count = 0
        labels = []
        for line in fin.readlines():
            try:
                rna, label = line.split('\t')
            except:
                print("ERROR")
                break
            doc += rna + ' ' 
            labels.append(label)
        #fout.write(doc)
        ws = WordSegment(doc,max_word_len=6,min_aggregation=1.0,min_entropy=0.6)
        res = ws.segSentence(doc)
        length = 0
        vector = [0 for i in range(ws.N)]
        for item in res:
            if(item  == ' '):
                continue
            length += len(item)
            if(length < ws.rna_len):
                if item in ws.words:
                    vector[ws.words.index(item)] += 1
                #fout.write(item+' ')
            else:
                #print(res[i+1])
                #fout.write(item+'\t')
                for i in range(len(vector)):
                    if i == 0:
                        fout.write(str(vector[i]))
                    else:
                        fout.write(' '+str(vector[i]))
                length = 0
                fout.write('\t'+labels[count])
                count += 1
                vector = [0 for i in range(ws.N)]
        
        for item in ws.word_with_freq:
            fwords.write(item[0]+'\t'+str(item[1])+'\n')
        fin.close()
        fout.close()
        fwords.close()

if __name__ == '__main__':
    segment_data()
