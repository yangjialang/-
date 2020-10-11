import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv("C:\\Users\\forever\\Desktop\\new_data.csv")
data_set = df.values

x = data_set[:, 5:20]
y = data_set[:, 4]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1, stratify=y)
ans = []

df1 = pd.read_csv("C:\\Users\\forever\\Desktop\\test_set.csv")
test_set = df1.values
x_t = test_set[:, 5:20]
y_t = test_set[:, 4]

# svm
time_start=time.time()
SVM = SVC()
SVM.fit(x_train, y_train)
y_predict_SVM = SVM.predict(x_t)
SVM_accuracy = accuracy_score(y_t, y_predict_SVM)
SVM_accuracy = round(SVM_accuracy, 4)
time_end=time.time()
ans.append(("Svm", SVM_accuracy, 'running time',time_end-time_start))

# DT
time_start=time.time()
DT = DecisionTreeClassifier()
DT_para = DT.get_params()
DT.fit(x_train, y_train)
y_predict_DT = DT.predict(x_t)
DT_accuracy = accuracy_score(y_t, y_predict_DT)
DT_accuracy  = round(DT_accuracy, 4)
time_end=time.time()
ans.append(("Decision tree", DT_accuracy , 'running time',time_end-time_start))

# knn
time_start=time.time()
KNN = KNeighborsClassifier()
KNN.fit(x_train, y_train)
y_predict_KNN = KNN.predict(x_t)
KNN_accuracy = accuracy_score(y_t, y_predict_KNN)
KNN_accuracy = round(KNN_accuracy, 4)
time_end=time.time()
ans.append(("K-nn", KNN_accuracy, 'running time',time_end-time_start))

# MLP
time_start=time.time()
MLP = MLPClassifier()
MLP.fit(x_train, y_train)
y_predict_MLP = MLP.predict(x_t)
MLP_accuracy = accuracy_score(y_t, y_predict_MLP)
MLP_accuracy = round(MLP_accuracy, 4)
time_end=time.time()
ans.append(("Mlp", MLP_accuracy, 'running time',time_end-time_start))

print(ans[0], "\n", ans[1], "\n", ans[2], "\n", ans[3])