from sklearn import svm
import sklearn
from code.feature_engineering import get_data_sep
from sklearn.model_selection import train_test_split

protein_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
                'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                'LIN28B',
                'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
                'WTAP', 'ZC3H7B']

data, dlabels = get_data_sep(data_path="../../../dataset/RNA_trainset/")
for (i, rnas, labels) in zip(range(37), data, dlabels):
    X_train, X_test, y_train, y_test = train_test_split(rnas, labels, test_size=0.2, random_state=42)
    model = svm.SVC(kernel='sigmoid')
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    score = sklearn.metrics.accuracy_score(y_test, predicted)
    print(protein_list[i] + " acc " + str(score))
    #fout.write(protein_list[i] + "acc" + score)
