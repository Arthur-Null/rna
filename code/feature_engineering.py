import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import time
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


def get_data_sep(data_path="../dataset/RNA_trainset/", positive=1, negative=0):
    """
    rarely same as above
    """
    rnas = []
    labels = []
    encoder = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
    for pot in protein_list:
        print('-' * 20 + pot + '-' * 20)
        replicate = set()
        fin = open(data_path + pot + '/train')
        X = []
        y = []
        for line in fin.readlines():
            rna, label = line.split('\t')
            if rna in replicate:
                continue
            else:
                replicate.add(rna)
            label = positive if int(label) == 1 else negative
            rna = list(map(lambda x: encoder[x], rna))
            X.append(rna)
            y.append(label)
        enc = OneHotEncoder(n_values=4)
        X = enc.fit_transform(X)
        rnas.append(X)
        labels.append(y)
    return rnas, labels


def get_data_2(data_path="../dataset/RNA_trainset2/", positive=1, negative=0, unwatched=-1):
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
    all_seq = []
    energies = []

    for iter_protein in range(number_of_protein):
        fin = open(data_path + protein_list[iter_protein] + '/train', 'r')
        fin_2nd = open(data_path + protein_list[iter_protein] + '/train_2nd', 'r')
        fin_lines = fin.readlines()
        fin_2nd_lines = fin_2nd.readlines()
        length = len(fin_lines)
        line2 = None
        for i in range(length):
            try:
                line = fin_lines[i]
                line2 = fin_2nd_lines[2 * i]
                label2 = fin_2nd_lines[2 * i + 1]

                rna, label = line.split('\t')

                line2 = line2.replace("\n", "")
                line2 = line2.split(' ')
                line2 = [i for i in line2 if i != "" and i != "(" and i != ")"]
                seq, energy = line2
                energy = energy.replace('(', "")
                energy = energy.replace(')', "")
                energy = float(energy)

                label = int(label)
                label2 = int(label2)

                assert label == label2
            except Exception as e:
                print(line2)
                print(e)
                break

            if rna not in rna_dic:
                rna_dic[rna] = len(all_rna)
                all_rna.append(rna)
                all_seq.append(seq)
                energies.append(energy)
                labels.append(np.ones([number_of_protein]) * unwatched)

            label = int(label)
            labels[rna_dic[rna]][iter_protein] = positive if label == 1 else negative
    labels = np.array(labels)
    assert labels.shape == (len(all_rna), number_of_protein)
    return all_rna, all_seq, labels, energies

def cal_accuracy(label, pred, thethold=0.5):
    aucs = []
    for i in range(len(label)):
        total = 0.
        match = 0.
        for j in range(len(label[i])):
            if label[i][j] != -1:
                total += 1
                if label[i][j] == 0 and pred[i][j] < thethold or label[i][j] == 1 and pred[i][j] >= thethold:
                    match += 1
        aucs.append(match / total)
    return aucs



if __name__ == '__main__':
    start = time.time()
    rnas, all_seq, labels, energies = get_data_2()
    end = time.time()
    print(len(rnas[1]), len(labels[1]), end-start, len(all_seq), len(energies), len(rnas))

