import os
import numpy as np

protein_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
                'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                'LIN28B',
                'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
                'WTAP', 'ZC3H7B']

number_of_protein = len(protein_list)


def get_data(data_path="../dataset/RNA_trainset/", positive=1, negative=0, unwatched=-1):
    """
    Get data from dataset.
    Input:
        data_path: The path of dataset
        positive: The expectation label of positive labels
        negative: The expectation label of negative labels
        unwatched: The expectation label of unwatched labels

    Output:
        all_rna: A list which contains all rna sequences.
        labels: A numpy array with shape [len(all_rna), number_of_protein]
    """
    rna_dic = {}
    all_rna = []
    labels = []

    for iter_protein in range(number_of_protein):
        fin = open(data_path + protein_list[iter_protein] + '/train', 'r')
        for line in fin.readlines():
            try:
                rna, label = line.split('\t')
            except:
                print("ERROR")
                break

            if rna not in rna_dic:
                rna_dic[rna] = len(all_rna)
                all_rna.append(rna)
                labels.append(np.ones([number_of_protein]) * unwatched)

            label = int(label)
            labels[rna_dic[rna]][iter_protein] = positive if label == 1 else negative
    labels = np.array(labels)
    assert labels.shape == (len(all_rna), number_of_protein)
    return all_rna, labels


if __name__ == '__main__':
    get_data()
