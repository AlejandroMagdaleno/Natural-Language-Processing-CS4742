import requests
import scipy as scipy
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
import fasttext
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



def get_article(url):
    page = requests.get(url)

    soup = BeautifulSoup(page.content, 'html.parser')
    body = soup.find_all(class_='cnnBodyText')[2].get_text()

    with open("news.txt", 'w') as f:
        f.write(body)


def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


if __name__ == '__main__':
    # get_article('url of cnn article')
    nlp = spacy.load("en_core_web_sm")

    out = []
    seen = set()

    with open('news.txt') as f:
        lines = f.read()
        doc = nlp(lines)

        num_of_sentences = sent_tokenize(lines)

        for token in doc:
            if token.is_punct is False and token not in seen:
                out.append(token.text)
            seen.add(token.text)

    model = fasttext.train_unsupervised('news.txt', model='skipgram')

    print(len(num_of_sentences))
    print(num_of_sentences[2])


    print(model['welcome'])
    print(cos_sim(model['spending'], model['america']))

    vectorizer = TfidfVectorizer()
    vectorizer.fit([lines])

    cv = CountVectorizer(ngram_range=(5, 5), stop_words='english')
    X = cv.fit_transform(lines)
    Xc = (X.T * X)
    Xc.setdiag(0)