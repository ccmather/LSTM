import codecs
import numpy as np
import jieba
from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
from keras.utils import to_categorical
import yaml
from keras.preprocessing import sequence
from sklearn import metrics
import nltk

#set parameters:
vocab_dim = 40
n_iterations = 1
n_exposures = 10
maxlen = 100
input_length = 100
batch_size = 8
n_epoch=1

def loadfile():
    creditcard = []
    gjdcard = []
    with codecs.open('home/corpus/positive.csv','r','utf-8') as fi1, codecs.open('corpus/negitive.csv', 'r', 'utf-8') as fi2:
       lines1 = fi1.readlines()
       lines2 = fi2.readlines()
       for line in lines1:
           creditcard.append(line)
       for line2 in lines2:
           gjdcard.append(line2)

    combined = np.concatenate([creditcard,gjdcard])
    cred_array = np.array([-1]*len(creditcard), dtype=int)
    gjd_array = np.array([0] * len(gjdcard), dtype=int)
    y = np.hstack([cred_array, gjd_array])
    print(combined.size)
    return combined,y

# 分词，方便期间，选择结巴分词，可适当替换为其他分词工具
def segment(document):
    result_list = []
    for text in document:
        result_list.append(' '.join(jieba.cut(text)).strip())
    print(result_list)
    return result_list


# 读取word2vec已经训练好的词向量模型，因为.mod文件含特殊字符，需要解码
def getfile(corpus):
    list = []
    w2index = dict()
    inx = 0
    word2vec = dict()
    with codecs.open('home/word2vec.mod','r','utf8') as fi:
        list = fi.readlines()
        for line in list:
            info = line.strip().split('=@')
            info[0]=info[0].encode('utf8').decode('utf-8-sig')   #去除特殊编码字符
            word2vec[info[0]] = info[1]
            w2index[inx] = info[0]
            inx=inx+1

        print(w2index)


    def parse_dataset(corpus):
        data = []
        for sentence in corpus:
            new_txt = []
            sentence = sentence.strip().split(' ')
            for word in sentence:
                for k in w2index:
                    if w2index[k]==word:
                        try:
                            new_txt.append(k)
                        except:
                            new_txt.append(0)
            data.append(new_txt)
        print(data)
        return data
    corpus = parse_dataset(corpus)
    corpus = sequence.pad_sequences(corpus, maxlen=maxlen)
    print(w2index,word2vec,corpus)
    return w2index,word2vec,corpus

def get_data(w2index,word2vec,combined,y):
    n_symbol = len(w2index)+1
    print(n_symbol)
    embedding_weight = np.zeros((n_symbol,vocab_dim))
    for k,v in w2index.items():
        value = word2vec[v].split(',')
        if len(value)<vocab_dim:
            continue
        print(len(value))
        embedding_weight[k,:] = value
    x_train, x_test,y_train,y_test = train_test_split(combined,y,test_size=0.2)
    y_train = to_categorical(y_train,num_classes=3)
    y_test = to_categorical(y_test,num_classes=3)
    return n_symbol,embedding_weight,x_train,y_train,x_test,y_test

def train_lstm(n_symbol,embedding_weights,x_train,y_train,x_test,y_test):
    print("Defining a simple keras Model…………")
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbol,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(output_dim=50,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3,activation='softmax'))
    model.add(Activation('softmax'))

    print('Compiling the Model……')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("train……………………")
    model.fit(x_train,y_train,batch_size=batch_size, epochs=n_epoch,verbose=1)

    print("Evaluate……………………")
    score = model.evaluate(x_test,y_test,batch_size=batch_size)



    yaml_string = model.to_yaml()
    with codecs.open('home/model/lstm.yml','w') as outfile:
        outfile.write(yaml.dump(yaml_string,default_flow_style=True))
    model.save_weights('home/model/lstm.h5')
    print('Test Score:',score)


if __name__=='__main__':
    combined, y = loadfile()
    result = segment(combined)
    w2idx,w2vec,combined = getfile(result)
    n_symbol, embedding_weight, x_train, y_train, x_test, y_test=get_data(w2idx,w2vec,combined,y)
    train_lstm(n_symbol,embedding_weight,x_train,y_train,x_test,y_test)