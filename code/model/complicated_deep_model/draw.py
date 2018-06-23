import os
import numpy as np
import matplotlib.pyplot as plt

pot_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
            'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
            'LIN28B',
            'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
            'WTAP', 'ZC3H7B']


def get_auc(pot):
    for path in os.walk("conv2d_models/" + pot + '/'):
        i = 0
        while path[1][i] == "checkpoint":
            i = i + 1
        file = open("conv2d_models/" + pot + '/' + path[1][i] + "/log")
        result = -1
        for line in file:
            if line[:2] == "Tr":
                continue
            auc = float(line.split(" ")[-1])
            result = max(result, auc)
        return result


if __name__ == '__main__':
    aucs = [i for i in map(get_auc, pot_list)]
    print(np.mean(aucs))
    plt.bar(np.arange(37), aucs)
    plt.show()

