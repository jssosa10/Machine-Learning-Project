import shogun as sh
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
"""
	funicones extra
"""
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
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
	Procesar los datos de entrada solo quedarse con las noticias relevantes y dividir el dataset asi
	us + world = 1
	sport + entertainment = 0
"""
with open('news.txt') as f:
	news = [line  for line in f]
with open('labels.txt') as f:
	labels = [line for line in f]
x = []
y = []
xy = []
news = [e.replace('\n', '').replace('\r', '') for e in news]
for i in range(len(news)):
	if news[i] is '':
		continue
	if 'us' in labels[i] or 'world' in labels[i]:
		xy.append((news[i],1))
	elif 'sport' in labels[i] or 'entertainment' in labels[i]:
		xy.append((news[i],-1))
np.random.shuffle(xy)
for w in xy:
	x.append(w[0])
	y.append(w[1])
tam = int(len(x)*0.8)
x_train = x[0:tam]
y_train = y[0:tam]
x_test = x[tam+1:len(x)]
y_test = y[tam+1:len(x)]
"""
	Teniendo en cuenta los resultados enteriores se obtuvo que 
	clase 1 = 16405
	calse 0 = 11476
	probablemente sera util hacer un oversample.

"""
"""
	Clasificacion con bolsa de palabras y kernel gaussiano 'rbf' fue necesario fijar el gamma
"""
C_range = [0.5, 1.0, 1.5, 1.75, 2]
gamma_range = [0.2,0.5,1, 1.5, 2]
#C_range = [1.0, 1.5]
#gamma_range = [0.5,1,1.5]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
param_grid = {'svm__gamma': gamma_range,
                     'svm__C': C_range}
pipe = Pipeline([('vect',stemmed_count_vect),('tfidf',TfidfTransformer()),('svm',SVC())])
print pipe.get_params().keys()
grid = GridSearchCV(pipe, cv=3, n_jobs=3, param_grid=param_grid)
grid.fit(x_train, y_train)
print("The best parameters are %s with a score of %0.3f"
      % (grid.best_params_, grid.best_score_))
means = grid.cv_results_['mean_test_score']
for mean, params in zip(means, grid.cv_results_['params']):
    print("%0.3f for %r"
            % (mean, params))
y_true, y_pred = y_test, grid.predict(x_test)
print np.mean(y_pred==y_true)
"""
text_svm_clf = text_svm_clf.fit(x_train,y_train)
print 'Predecir'
prediction = text_svm_clf.predict(x_test)
print np.mean(prediction==y_test)
"""
