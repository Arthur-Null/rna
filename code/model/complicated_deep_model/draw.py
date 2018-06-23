import os
import numpy as np
import matplotlib.pyplot as plt

pot_list = ['AGO1', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
            'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
            'LIN28B',
            'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
            'WTAP', 'ZC3H7B']


def get_auc(pot):
    for path in os.walk("cov1d/" + pot + '/'):
        print(path)
        i = 0
        while path[1][i] == "checkpoint":
            i = i + 1
        file = open("cov1d/" + pot + '/' + path[1][i] + "/log")
        result = -1
        epoch = 0
        max_epoch = 0
        for line in file:
            if line[:2] == "Tr":
                continue
            if line == '':
                continue
            auc = float(line.split(" ")[-1])
            if auc > result:
                max_epoch = epoch
            result = max(result, auc)
            epoch += 1
        print(max_epoch)
        return result


if __name__ == '__main__':
    aucs = [i for i in map(get_auc, pot_list)]
    print(aucs)
    print(np.mean(aucs))
    plt.bar(np.arange(36), aucs)
    plt.show()

