import os, shutil
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import pickle
import fasttext

def extract(i):
    file_path = './test/'+i

    sub = []
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

        final = ""

        try:
            sub = sub[0]
            sub = sub.lower()
            sub = sub.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip().split()


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

        return final


def main():
    categories = int(input('Please type number of categories (Optimal value is 4):'))

    with open('k-means-clusters-'+str(categories)+'.pickle', 'rb') as handle:
        kmeans = pickle.load(handle)
    try:
        shutil.rmtree('./test/categorize')
    except:
        pass

    os.makedirs('./test/categorize/')

    for i in range(categories):
        os.makedirs('./test/categorize/'+str(i))

    files = set(os.listdir('./test/'))-{'categorize'}

    for i in files:
        line = extract(i)
        vec = model.get_sentence_vector(i.strip())
        folder = kmeans.predict([vec])
        os.rename("./test/"+i, "./test/categorize/"+str(folder[0])+'/'+i)

if __name__ == '__main__':
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    model = fasttext.load_model("./crawl-300d-2M-subword.bin")
    main()
