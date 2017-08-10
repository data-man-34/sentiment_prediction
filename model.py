
import numpy as np
import nltk
import nltk.sentiment
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


np.random.seed(1)
# load data
amazon = open("amazon_cells_labelled.txt", 'r', encoding = 'utf8').readlines()
imdb = open("imdb_labelled.txt", 'r', encoding = 'utf8').readlines()
yelp = open("yelp_labelled.txt", 'r', encoding = 'utf8').readlines()
text = amazon + imdb + yelp
# make sentence, label lists
sentence = []
label = []
for t in text:
    t = t.replace('\n', '')
    s, l = t.split('\t')
    sentence.append(s)
    label.append(int(l))
# n_posts
n_posts = len(text)
'''
 LSTM model
'''
# connect sentence to string
sentence_str = ' '.join(sentence)
# term frequency dictionary
fdict = nltk.FreqDist(sentence_str.split())
# take words into a list
words = []
for term, f in fdict.items():
    words.append(term)
words = sorted(words)
# make word-int mapping
word2int = {}
for i in range(len(words)):
    word2int[words[i]] = i + 1
# convert sentence to int sequence
sentence_index = []
for t in sentence:
    sentence_index.append([word2int[w] if w in word2int.keys() else 0 for w in t.split()])
# fill all sentences with '0' to make sentences equal lengths
maxLength = 0
for t in sentence_index:
    if maxLength < len(t):
        maxLength = len(t)
sentence_index_filled = []
for t in sentence_index:
    sentence_index_filled.append([0] * (maxLength - len(t)) + t)
# make data
x = np.reshape(sentence_index_filled, (n_posts, maxLength))
y = np.reshape(label, (n_posts, 1))
# sample 20% for testing
test_id = np.random.choice(x.shape[0], int(x.shape[0] * 0.2), replace = False)
train_id = np.array([i for i in list(range(x.shape[0])) if i not in test_id])
train_x = x[train_id,:]
train_y = y[train_id,:]
test_x = x[test_id,:]
test_y = y[test_id,:]
# define model structure
model = Sequential()
model.add(Embedding(len(words), 200, input_length=maxLength))
model.add(LSTM(100))
model.add(Dropout(0.1))
model.add(Dense(1, activation = "sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# define the checkpoint
filepath="sentiment-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(train_x, train_y, epochs=20, batch_size=50, callbacks = callbacks_list)

# make prediction
rnn_pre = model.redict(test_x)
'''
 Vader model
'''
# declare Sentiment Intensity Analyzer object
obj = nltk.sentiment.vader.SentimentIntensityAnalyzer()
# find good cut-off
for i in np.arange(0, 1, 0.1):
    pre = []
    for s in np.array(sentence)[test_id]:
        temp = obj.polarity_scores(s)['compound']
        temp = 0 if temp < i else 1
        pre.append(temp)
    
    print("cut:", i, "acc:", np.sum(np.array(pre) == test_y.reshape(-1))/test_y.shape[0])
# use the best cut-off for prediction
vader_pre = []
for s in np.array(sentence)[test_id]:
    temp = obj.polarity_scores(s)['compound']
    temp = 0 if temp < 0.1 else 1
    vader_pre.append(temp)

'''
 Ensemble modeling
'''
best_model = load_model('sentiment-00-0.3790.hdf5')
rnn_pre = best_model.predict(test_x)
rnn_pre = rnn_pre.reshape(-1)
predict = []
for i in range(test_y.shape[0]):
    if rnn_pre[i] > 0.5 and vedar_pre[i] == 1:
        predict.append(1)
    elif rnn_pre[i] < 0.5 and vedar_pre[i] == 0:
        predict.append(0)
    else:
        if rnn_pre[i] > 0.9:
            predict.append(1)
        elif rnn_pre[i] < 0.1:
            predict.append(0)
        else:
            predict.append(vedar_pre[i])

print(np.sum(np.array(predict) == test_y.reshape(-1))/test_y.shape[0])