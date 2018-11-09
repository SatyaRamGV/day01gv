

```python
from random import randint
import numpy as np
import pandas as pd
import random
import json
import torch
import os
import re

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from nltk.corpus import stopwords

from models import InferSent
use_cuda = True
%matplotlib inline
```


```python
path = os.getcwd()
data = pd.read_csv(f'{path}/gcp_01.csv')
data = data[["Category", "Sub Category", "Questions", "Answers"]].dropna()
glove_path = '/home/glove.840B.300d.txt' 
```


```python
cat = data['Category'].tolist()
ques = data['Questions'].tolist()
ans = data['Answers'].tolist()
```


```python
stop_words = list(stopwords.words("english"))

for x in ['because','when','where','what','why','how']:
    stop_words.remove(x)
    
def text_clean(raw_text):
    letters_only = re.sub("[^a-zA-Z0-9]", " ", raw_text)
    words = letters_only.lower().split()                                        
    meaningful_words = [w for w in words if not w in stop_words]   
    return( " ".join(meaningful_words))
```


```python
ques_cleaned = []
for q in ques:
    ques_cleaned.append(text_clean(q))
```


```python
model_version = 1
MODEL_PATH = "infersent%s.pkl" % model_version                                    #specify the model path and version
params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
              'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}   #model parameters
model = InferSent(params_model)                                                   # set the model parameters
model.load_state_dict(torch.load(MODEL_PATH))                                     # load the pertrained model
model = model.cuda() if use_cuda else model                                       # model to cuda
W2V_PATH = glove_path                                                             # path for glove
model.set_w2v_path(W2V_PATH)                                                      # set the path of glove pretrained model
model.build_vocab_k_words(K=10000)                                                # Load Vocabulary from the Embeddings
model.update_vocab(ques)
embeddings = model.encode(ques_cleaned, bsize=64, tokenize=False, verbose=True)
print('nb sentences encoded : {0}'.format(len(embeddings))) 
```
-----

   Vocab size : 10000
   
   Found 185(/247) words with w2v vectors
   
   New vocab size : 10188 (added 185 words)
   
   Nb words kept : 2663/2852 (93.4%)
   
   Speed : 3507.3 sentences/s (gpu mode, bsize=64)
   
   nb sentences encoded : 288
   
-----


```python
context = {}

ERROR_THRESHOLD = 0.4

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def get_similar_questions(query_processed, num_results = 3, question_embeddings = embeddings, df = data):
    
    encoded_ques = model.encode([query_processed])[0]
        
    question_embeddings = embeddings

    # the metric we used here is cosine, the cosine distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
    pairwise_dist = pairwise_distances(question_embeddings, encoded_ques.reshape(1,-1), metric= 'cosine')
    
    # np.argsort will return indices of the smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]

    #pdists will store the smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
    
    return indices, pdists
```


```python
get_similar_questions('what is an unexpected Adverse Drug Reaction')
```

   (array([ 59,   0, 170]),
    array([0.10569298, 0.1756931 , 0.35103738], dtype=float32))


```python
context = {}

ERROR_THRESHOLD = 0.4

def classify(sentence):
    
    # generate probabilities from the model to predict the category
    results = model.predict([bow(sentence, words)])[0]
    
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r > ERROR_THRESHOLD]
    
    # sort by strength of probability
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list
```


```python
def response(query, num_results = 3, return_imp = False, visualize = False):
    
    if query in ['Bye', 'bye', 'Good Bye', 'Thank you']:
        answers = 'Bye!'
    else:
        query_processed = text_clean(query)
        indices, distances = get_similar_questions(query_processed, num_results)
        
        for i in range(0,len(indices)):
            print('\n ___{}: {}'.format(i, ans[indices[i]]))
        
    if visualize == True:
        _, _ = model.visualize(query_processed)
        
     #InferSent importances
    if return_imp == True:
        idx, imp_score = model.get_importance(query_processed)
        zipped = zip(query_processed.split(), imp_score[1:-1])
        word_imp = sorted(zipped, key = lambda x: x[1])
        print('\n ___ANSWER DISTANCES:', distances)
        print('\n ___WORD IMPORTANCE:', word_imp)
```


```python
# response('what is an unexpected Adverse Drug Reaction?', return_imp = False)
```


```python
print('BOT:', 'Hi I am Clinical Bot, how can I help you?')

while True:
    print('\n')
    query = str(input())
    
    if query in ['Bye', 'bye', 'Good Bye', 'Thank you']:
          break
    else: response(query, return_imp = True)
```
---

BOT: Hi I am Clinical Bot, how can I help you.



Bye!!!

---

