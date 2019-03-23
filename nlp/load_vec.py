from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
from os.path import basename, normpath, exists
from os import stat
import re
import string
from progress.bar import ChargingBar
import time

def obtainModel(path):
#    file = basename(normpath(path))
    file = path + '.prep' # saving in the same folder
    if exists(file):
        return KeyedVectors.load(file)
    else:
        model = KeyedVectors.load_word2vec_format(path, binary=False)
        model.save(file)
        return model

# returns path to a new file with the preprocessed corpus text within
def preprocessCorpus(path):
    file = path + '.prep' # saving in the same folder
    if exists(file):
        return file
        
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
        byteCounter = 0
        mbSize = 1024**2
        size = int(stat(path).st_size/mbSize)
        bar = ChargingBar('Preprocessing', max=size)
        for lineStr, lengthStr in byLines(path):
            if lineStr:
                f.write(lineStr + '\n')
            byteCounter += lengthStr
            if int(byteCounter / mbSize) > 0:
                bar.next()  # progress is updated by megabytes
                byteCounter %= mbSize
        bar.finish()                
    return file

model = obtainModel('../../llearn/data/104/model.txt')
corpusFile = preprocessCorpus('../../llearn/data/norsk_aviskorpus/1/19981013-20010307/alle-981013-010307.utf8')

if 'lære' in model:
    print(model['lære'].shape)
else:
    print('{0} is an out of dictionary word'.format('lære'))

# Some predefined functions that show content related information for given words
print(model.most_similar(positive=['kvinne', 'konge'], negative=['mann']))
