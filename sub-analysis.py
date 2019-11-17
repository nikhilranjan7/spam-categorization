import os
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import fasttext
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter, defaultdict
from sklearn.cluster import DBSCAN

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

#Length of dataset = 10219

senten = []
count_loop = 0
for i in os.listdir('./email/'):
    sub = []
    # if count_loop == 100:
    #     break
    count_loop += 1
    #print(i)
    if os.path.isdir('./email/'+i) == False:
        with open('./email/'+i, 'r', encoding='utf8', errors='ignore') as f:
            l = f.readlines()

        words = []
        aftersub = False
        for j in range(len(l)):
            if aftersub == False and 'Subject:' in l[j][:11]:
                sub.append(l[j][9:])
                aftersub = True

            elif aftersub == True:
                try:
                    decoded = base64.b64decode(l[j])
                    for k in decoded.strip().split():
                        if k.isalpha():
                            words.append(k)

                except:
                    for k in l[j].strip().split():
                        if k.isalpha():
                            words.append(k)


    else: #if-directory
        for k in os.listdir('./email/'+i+'/'):
            with open('./email/'+i+'/'+k, 'r', encoding='utf8') as f:
                l = f.readlines()

            aftersub = False
            for j in range(len(l)):
                if aftersub == False and 'Subject:' in l[j][:11]:
                    sub.append(l[j][9:])
                    aftersub = True

                elif aftersub == True:
                    try:
                        decoded = base64.b64decode(l[j])
                        for k in decoded.strip().split():
                            if k.isalpha():
                                words.append(k)

                    except:
                        for k in l[j].strip().split():
                            if k.isalpha():
                                words.append(k)
    try:
        sub = sub[0]
        sub = sub.lower()
        sub = sub.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip().split()

        final = ""

        for w in sub:
            w = w.lower()
            w = lemmatizer.lemmatize(w)
            if w not in stop_words:
                final += w
                final += ' '

    except:
        pass

    for w in words:
        if type(w) == bytes:
            w = w.decode()
        w = w.lower()
        w = lemmatizer.lemmatize(w)
        if w not in stop_words:
            final += w
            final += ' '

    #print(final)
    senten.append(final)

len(senten)
senten[:20]

import pickle

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)


model = fasttext.load_model("./crawl-300d-2M-subword.bin")

vecs = []
for i in senten:
    vecs.append(model.get_sentence_vector(i.strip()))

with open('sent-embeddings.pickle', 'wb') as handle:
    pickle.dump(vecs, handle, protocol=pickle.HIGHEST_PROTOCOL)

from sklearn.cluster import KMeans
import numpy as np

X = np.array(vecs)

for i in range(2,11):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    with open('k-means-clusters-'+str(i)+'.pickle', 'wb') as handle:
        pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    c = Counter(kmeans.labels_)
    print([(i, c[i] / len(kmeans.labels_) * 100.0) for i in c])


clustering = DBSCAN().fit(X)
len(clustering.labels_)
arr = clustering.fit_predict(X)
uni, counts = np.unique(arr, return_counts=True)
d = dict(zip(uni, counts))
print (d)

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
first, second, third, fourth = kmeans.cluster_centers_[0], kmeans.cluster_centers_[1], kmeans.cluster_centers_[2], kmeans.cluster_centers_[3]

euclidean_distances([first], [second])


for j in vecs[:20]:
    d = [euclidean_distances([j], [first])[0][0], euclidean_distances([j], [second])[0][0], euclidean_distances([j], [third])[0][0], euclidean_distances([j], [fourth])[0][0]]
    print(d.index(min(d))+1)

senten[:20]


lemmatizer.lemmatize('longer')
type(b'op')

count = 0
for i in range(len(sub)):
    if '=?' in sub[i][:5]:
        count+=1
print(count)

sub[:10]


with open('/Users/e102947/Desktop/Hackathon/email/1562150311.M280237P18659.mail1-fi1.d-fence.eu,S=9672,W=9815', 'r', encoding='utf8') as f:
    l = f.readlines()

l[43]

import base64

base64.b64decode(l[51])
'Content-Transfer-Encoding: asfsf'[27:]

'abc:'.isalpha()
