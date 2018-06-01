import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

protein_list = ['AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1', 'DGCR8', 'EIF4A3', 'EWSR1',
                'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2', 'IGF2BP3', 'LIN28A',
                'LIN28B',
                'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43', 'TIA1', 'TIAL1', 'TNRC6', 'U2AF65',
                'WTAP', 'ZC3H7B']

encoder = {'A': 0, 'G': 1, 'C': 2, 'T': 3}
fout = open('result', 'w')

for pot in protein_list:
    replicate = []
    print('-' * 20 + pot + '-' * 20)
    fin = open('../../../dataset/RNA_trainset/' + pot + '/train')
    X = []
    y = []
    for line in fin.readlines():
        rna, label = line.split('\t')
        if rna in replicate:
            continue
        else:
            replicate.append(rna)
        label = int(label)
        rna = list(rna)
        for i in range(len(rna)):
            rna[i] = encoder[rna[i]]
        X.append(rna)
        y.append(label)
    #print(X[0])
    enc = OneHotEncoder(n_values=4)
    X = enc.fit_transform(X)
    #print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_jobs=-1, n_estimators=10000, max_depth=4, learning_rate=5e-3)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)], eval_metric=['auc', 'error', 'logloss'], early_stopping_rounds=50)
    # pred = model.predict_proba(X_test)
    # auc = roc_auc_score(y_test, pred[:, 1])
    # loss = log_loss(y_test, pred[:, 1])
    index = model.best_iteration
    auc = model.evals_result()['validation_1']['auc'][index]
    loss = model.evals_result()['validation_1']['logloss'][index]
    error = model.evals_result()['validation_1']['error'][index]
    fout.write(pot + "\tAUC:{:.4f}\tLogLoss:{:.4f}\tError:{:.4f}\n".format(auc, loss, error))
    print('-' * 16 + pot + ' finish' + '-' * 16)
