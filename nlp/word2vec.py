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

#if 'lære' in model:
#    print(model['lære'].shape)
#else:
#    print('{0} is an out of dictionary word'.format('lære'))

# Some predefined functions that show content related information for given words
#print(model.most_similar(positive=['kvinne', 'konge'], negative=['mann']))
        
article_words_count = 5000
article = sampleSequentialLines(corpus_path, corpus_lines, lines_to_read=article_words_count)
chosen_words = random.sample(article, int(article_words_count * 0.1))
y = [1 if el in chosen_words else 0 for el in article]
non_existent_word = zeros(model.vectors[0].shape)
X = [model[w] if w in model else non_existent_word for w in article]

evaluateOnData(X, y, kernel=['rbf'], gamma=[100000])

