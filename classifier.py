import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

data_dict = pickle.load(open('./data.pickle','rb'))

#NB i nostri dati sono il liste, quindi li convertiamo in np
data = np.asarray(data_dict['data']) #nel frattempo li converto in np
labels = np.asarray(data_dict['labels'])
#Ora abbiamo tutti i dati in questi array
#Divideremo questi dati in training set e test set, per allenare
#l'algoritmo e testare le performance

#ATTENZIONE: qui c'è la scelta del classificatore
#RANDOM FOREST
#Attenzione anche alle metriche, magari inserirne più di una

#dividiamo l'array in 2 set, train e test, stessa cosa per i labels
#in questo caso il test set è il 20% (0.2) comunemente usato
#shuffle=True mischiamo i dati, good practice
#Ci sono dei bias di cui non ci rendiamo conto quindi è meglio mischiare i dati
#stratify=labels, divideremo il dataset ma manterremo le stesse proporzioni per ogni label
# sia per training set sia per test set 
#NB: se devo fare test mettere almeno 2 immagini per ogni label altrimenti si inceppa stratify
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Questo è molto veloce, semplice
#model = RandomForestClassifier()

#Naive Bayes -> peggiore
#model = GaussianNB() 

#Support Vector Machines  -> migliore
model = SVC()

#K-Nearest Neighbors  -> migliore 2
#model = KNeighborsClassifier()

#Decision Tree
#model = DecisionTreeClassifier();

#Logistic Regression  -> anche questo molto bene
#model = LogisticRegression()

#Reti neurali Multilayer Perceptron -> anche questo bene ma forse è troppo complicato, valutare uno dei precedenti
#model = MLPClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

f = open('model.p', 'wb') #da libreria pickle
pickle.dump({'model':model}, f)
f.close()

print('Modello salvato.')

#print(data_dict.keys())
#print(data_dict)