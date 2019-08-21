import numpy as np
import pandas as pd
# import xlrd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def pad_dict_list(dict_list, padel):
    lmax = 0
    for lname in dict_list.keys():
        lmax = max(lmax, len(dict_list[lname]))
    for lname in dict_list.keys():
        ll = len(dict_list[lname])
        if  ll < lmax:
            dict_list[lname] += [padel] * (lmax - ll)
    return dict_list

filename = 'data_SS_test.csv'
# wordbook = xlrd.open_workbook(filename=file)
#print(wordbook.sheet_names())
# sheet1 = wordbook.sheet_by_index(0)

# cols0 = sheet1.col_values(0)
# cols1 = sheet1.col_values(1)

originData = pd.read_csv(filename, header = 0, sep = ',')
# print(originData)
trainData = np.array([x for x in originData['train'].values if str(x) != 'nan']).reshape(-1,1)
# print(trainData)
testData = np.array([x for x in originData['test'].values if str(x) != 'nan']).reshape(-1,1)
# print(testData)

clf = IsolationForest(behaviour='new', max_samples='auto', contamination=0.065)
clf.fit(trainData)

y_pred_train = clf.predict(trainData)
y_pred_test = clf.predict(testData)
data_dict = {}
data_dict['train_pred'] = y_pred_train.tolist()
data_dict['test_pred'] = y_pred_test.tolist()
# print(data_dict)
res = pd.DataFrame.from_dict(pad_dict_list(data_dict, ''))
res.to_csv('predict_res.csv')

# X_train = np.c_[cols0]
# print(X_train)
# X_text = np.c_[cols1]
#print(X_text)

# clf = IsolationForest(behaviour='new', max_samples='auto', contamination=0.065)
# clf.fit(X_train)
# y_pred_train = clf.predict(X_train)
# y_pred_test = clf.predict(X_text)
##y_pred_outliers = clf.predict(X_text)

# y_o = np.c_[y_pred_train,y_pred_test]
# print(y_pred_train)
# np.savetxt("y_o.txt",y_o)
