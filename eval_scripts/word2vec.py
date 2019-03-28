from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from os.path import basename, normpath, exists
from os import stat
import re
import string
from progress.bar import ChargingBar
import time
import eval_bin_classifier
import random
import ast
from itertools import islice
from numpy import zeros
from eval_bin_classifier import evaluateOnData
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from stop_words import get_stop_words
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def obtainModel(path):
#    file = basename(normpath(path))
    file = path + '.prep' # saving in the same folder
    if exists(file):
        return KeyedVectors.load(file)
    else:
        model = KeyedVectors.load_word2vec_format(path, binary=False)
        model.save(file)
        return model

# returns path to a new file with the preprocessed corpus text within and its line count 
def preprocessCorpus(path):
    file = path + '.prep' # saving in the same folder
    if exists(file):
        with open(file + '.cnt', 'r') as cf:
            s = cf.readline().strip()
            return file, int(s)
        
    # returns a new preprocessed string and length of the original string
    def byLines(path):
        cleanr = re.compile('<.*?>')
        with open(path, encoding="utf-8") as fp:
            for line in fp:
                tagless = re.sub(cleanr, '', line) # remove tags
#                prep = tagless.translate(str.maketrans('', '', string.punctuation + string.digits)) # remove punctuation
                prep = ''.join(ch if ch.isalpha() or ch==' ' else '' for ch in tagless) # not the fastest. but translate has no unicode str.letters 
                yield prep.strip().lower(), len(line.encode("utf-8")) # size of unicode string in bytes
                
    with open(file, 'w', encoding="utf-8") as f:
        line_counter = 0
        byteCounter = 0
        mbSize = 1024**2
        size = int(stat(path).st_size/mbSize)
        bar = ChargingBar('Preprocessing', max=size)
        for lineStr, lengthStr in byLines(path):
            if lineStr:
                f.write(lineStr + '\n')
                line_counter += 1
            byteCounter += lengthStr
            if int(byteCounter / mbSize) > 0:
                bar.next()  # progress is updated by megabytes
                byteCounter %= mbSize
        bar.finish()
        with open(file + '.cnt', 'w') as cf:
            cf.write(str(line_counter))
    return file, line_counter

# Returns the list of lines picked consequently with up to line_count elements if available starting from random line
def sampleSequentialLines(file_path, total_lines, lines_to_read=10):
    with open(file_path, encoding="utf-8") as f:
        # waterman's reservoir takes much longer compared to simple line counting
        start = random.randrange(total_lines)
        res = [s.strip() for n, s in islice(enumerate(f), start, start + lines_to_read)]
        return res

        
model = obtainModel('../../llearn/data/104/model.txt')
corpus_path, corpus_lines = preprocessCorpus('../../llearn/data/norsk_aviskorpus/1/19981013-20010307/alle-981013-010307.utf8')

# Some predefined functions that show content related information for given words
#print(model.most_similar(positive=['kvinne', 'konge'], negative=['mann']))
        
article_words_count = 7500#5000
article = sampleSequentialLines(corpus_path, corpus_lines, lines_to_read=article_words_count)
# Filtering out the stop-words because they are rarely chosen by a user, yet once having been chosen for the tests they give too biased statistics
for_sampling = [x for x in article if x not in get_stop_words('norwegian')]
chosen_words = random.sample(for_sampling, int(article_words_count * 0.1))
# Removing words not in model since they don't add any useful information to the nn
X = [w for w in article if w in model]
y = [1 if el in chosen_words else 0 for el in article if el in model]

print ("Initial article size %i; Preprocessed article size %i" % (article_words_count, len(X)))

clf = evaluateOnData([model[x] for x in X], y, kernel=['rbf'], gamma=[ 1e-2]) # better generalisation
# clf = evaluateOnData([model[x] for x in X], y, kernel=['rbf'], gamma=[ 1e9]) # better fit

Xw_train, Xw_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1,random_state=109) # 80% training

count_eval = len(Xw_eval)
count_train = len(Xw_train)
print ("Evaluation vs. training word number: %i / %i" % (count_eval, count_train))

count_total_new = len([1 for v in Xw_eval if v not in Xw_train])
count_test_new = len([1 for i, v in enumerate(Xw_eval) if v not in Xw_train and y_eval[i]==1])
count_test_total = len([1 for i, v in enumerate(Xw_eval) if y_eval[i]==1])
print("New eval words vs. total eval words: %i / %i (%2.2f%%)" % (count_total_new, count_eval, 100*count_total_new/count_eval))
print("1-new eval words vs. 1-total eval words: %i / %i (%2.2f%%)" % (count_test_new, count_test_total, 100*count_test_new/count_test_total))

X_train = [model[w] for w in Xw_train]
X_eval = [model[w] for w in Xw_eval]

p = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)
y_pred = p.predict(X_eval)
train_wl = [model.most_similar(positive=[X_train[i]])[0] for i, _ in enumerate(y_train) if y_train[i]==1]
predicted_unexp = [model.most_similar(positive=[X_eval[i]])[0] for i, _ in enumerate(y_pred) if y_pred[i]==1 and y_eval[i]==0]
predicted_err = [model.most_similar(positive=[X_eval[i]])[0] for i, _ in enumerate(y_pred) if y_pred[i]==0 and y_eval[i]==1]
predicted_cmn = [model.most_similar(positive=[X_eval[i]])[0] for i, _ in enumerate(y_pred) if y_pred[i]==1 and y_eval[i]==1]

print('')
print ("Evaluation ...")
print('')
count_predict_cmn = len(predicted_cmn)
print("Predicted vs. planned: %i / %i (%2.2f%%)" % (count_predict_cmn, count_test_total, 100*count_predict_cmn/count_test_total))
count_predict_new = len([1 for v in predicted_cmn if v[0] not in Xw_train])
print("Predicted vs. planned (new words only): %i / %i (%2.2f%%)" % (count_predict_new, count_test_new, float('NaN') if count_predict_new == 0 else  100*count_predict_new/count_test_new))
print("   + %i words by virtue of generalisation" % len(predicted_unexp))

print ("Trained")
print ([v[0] for v in train_wl])

print ("Predicted correctly")
print ([v[0] for v in predicted_cmn])

print ("Mispredicted")
print ([v[0] for v in predicted_err])

print ("New prediction (based on generalisation)")
print ([v[0] for v in predicted_unexp])

# plot
X_tra = [v[0] for v in train_wl]
X_cmn = [v[0] for v in predicted_cmn]
X_err = [v[0] for v in predicted_err]
X_new = [v[0] for v in predicted_unexp]

plt.figure()
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Predicted/mispredicted/new word embeddings (via PCA)')

if len(X_tra):
    Xr = PCA(n_components=2).fit_transform([model[v] for v in X_tra])
    plt.scatter(Xr[:, 0], Xr[:, 1], color='yellow', label='Trained (Gaussian top)')
#    for label, x, y in zip(X_tra, Xr[:, 0], Xr[:, 1]):
#        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

if len(X_cmn):
    Xr = PCA(n_components=2).fit_transform([model[v] for v in X_cmn])
    plt.scatter(Xr[:, 0], Xr[:, 1], color='blue', label='Predicted')
    for label, x, y in zip(X_cmn, Xr[:, 0], Xr[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

if len(X_err):
    Xr = PCA(n_components=2).fit_transform([model[v] for v in X_err])
    plt.scatter(Xr[:, 0], Xr[:, 1], color='red', label='Mispredicted')
    for label, x, y in zip(X_err, Xr[:, 0], Xr[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

if len(X_new):
    Xr = PCA(n_components=2).fit_transform([model[v] for v in X_new])
    plt.scatter(Xr[:, 0], Xr[:, 1], color='green', label='New')
    for label, x, y in zip(X_new, Xr[:, 0], Xr[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

plt.legend(loc="best")
plt.show()

