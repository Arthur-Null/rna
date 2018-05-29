import os

pot_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
            'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A', 'LIN28B',
            'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
            'WTAP', 'ZC3H7B']

# for (root, dic, files) in os.walk('../dataset/RNA_trainset'):
#     pot = root.split('/')[-1]
#     pot_list.append(pot)
# pot_list.remove('RNA_trainset')
# pot_list.sort()

dic = {}
for pot in pot_list:
    fin = open('../dataset/')