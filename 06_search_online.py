import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import wikipedia

lemmatizer = nltk.stem.WordNetLemmatizer()

# only run once
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

wiki = wikipedia.page('List of Game of Thrones characters', auto_suggest=False).content

# get list of single sentences out of combined text
response = nltk.sent_tokenize(wiki)


def get_lemmas(text):
    # break text up into single words toLowerCase
    tokens = nltk.word_tokenize(text.lower())
    tags = nltk.pos_tag(tokens)

    text_lemmas = []

    for token, tag in zip (tokens, tags):
        # extract part-of-speech tag
        tag_pos = tag[1][0].lower()
        # exclude prepositions, articles, etc
        if tag_pos in ['n', 'v', 'a', 'r']:
            lemma = lemmatizer.lemmatize(token, tag_pos)
            text_lemmas.append(lemma)

    return text_lemmas


def find_similarity(response, query):
    # get lemmas out of list of sentences
    tv = TfidfVectorizer(tokenizer=get_lemmas)
    # generate matrix with weights for each lemma in the given text (how often do they appear)
    tf = tv.fit_transform(response)
    # Now we can calculate the relative similarity
    # of each sentence to the query string
    coefficients = cosine_similarity(tf[-1], tf)
    # so we need to extract the second to last
    index = coefficients.argsort().flatten()[-2]
    score = coefficients.flatten()[index]
    if score > 0:
        return ':: RESULT :: ' + response[index] + ' :: SCORE :: ' + str(score) + ' ::'
    else:
        return ':: INFO :: No Match Found'


while True:
    query = input(':: Query Input:: ')
    if query == 'quit':
        print(':: INFO :: Shutting down...')
        quit()
    else:
        response.append(query)
        output = find_similarity(response=response, query=query)
        print(output)