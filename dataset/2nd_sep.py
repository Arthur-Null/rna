pot_list = ['AGO1','AGO2','AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
            'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
            'LIN28B',
            'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
            'WTAP', 'ZC3H7B']

dic = {}
f = open('second_strcuture', 'r')
for line in f.readlines():
    line.replace('\n', '')
    rna, second = line.split('\t')
    if 'N' in rna:
        continue
    dic[rna] = second

for pot in pot_list:
    fin = open('./trainset/' + pot, 'r')
    fout = open('./trainset/' + pot + '_2nd', 'w')
    for line in fin.readlines():
        rna, label = line.split('\t')
        if 'N' in rna:
            continue
        second = dic[rna]
        fout.write(second[:-1] + '\t' + label)
