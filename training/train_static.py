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

#Metriche
from sklearn.metrics import classification_report, confusion_matrix

data_dict = pickle.load(open('../data/data.pickle','rb'))

#NB i nostri dati sono il liste, quindi li convertiamo in np
data = np.asarray(data_dict['data']) #nel frattempo li converto in np
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

#Support Vector Machines
model = SVC(probability=True)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)
print("Classification Report:")
print(classification_report(y_test, y_predict, target_names=np.unique(labels)))

# Salva anche le performance
with open('model_performance.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'accuracy': score,
        'y_test': y_test,
        'y_predict': y_predict,
        'labels': np.unique(labels)
    }, f)


print('{}% of samples were classified correctly!'.format(score * 100))

f = open('../models/model.p', 'wb') #da libreria pickle
pickle.dump({'model':model}, f)
f.close()

print('Modello salvato.')