import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import nltk
from sklearn.multiclass import OutputCodeClassifier
"""
Preparacion del stemmer
"""
#nltk.download()
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

""" 
	Procesar los datos de entrada solo quedarse con las noticias relevantes y dividir el dataset y crear un mapeo entre calse y valor
"""

with open('news.txt') as f:
	news = [line  for line in f]
with open('labels.txt') as f:
	labels = [line for line in f]
set_labels = set()
for x in labels:
    set_labels.add(x)
_labels_ = [l for l in set_labels]
dict_labels = {}
for i in range(len(_labels_)):
    dict_labels[_labels_[i]]=i
x = []
y = []
xy = []
news = [e.replace('\n', '').replace('\r', '') for e in news]
for i in range(len(news)):
	if news[i] is '':
		continue
	else:
		xy.append((news[i],dict_labels[labels[i]]))	
np.random.shuffle(xy)
for w in xy:
	x.append(w[0])
	y.append(w[1])
print(len(x))
tam = int(len(x)*0.8)
x_train = x[0:tam]
y_train = y[0:tam]
x_test = x[tam+1:len(x)]
y_test = y[tam+1:len(x)]

"""
	Entrenamiento Kernel Gaussiano multilabel one vs one
"""
C_range = [0.5,1.0, 1.5, 1.75, 2]
gamma_range = [0.2,0.5,0.75,1, 1.5, 2]
for C in C_range:
	for gamma in gamma_range:
		clf = Pipeline([('vect',stemmed_count_vect),('tfidf',TfidfTransformer()),('svm',SVC(C=C,gamma=gamma))])
		clf = clf.fit(x_train,y_train)
		y_true, y_pred = y_test, clf.predict(x_test)
		print (np.mean(y_pred==y_true),C,gamma)


