import nltk

# only run once
# nltk.download('wordnet')
# nltk.download('omw-1.4')

words = ['was', 'is', 'am', 'be']

lemmatizer = nltk.stem.WordNetLemmatizer()


for word in words:
    lemma = lemmatizer.lemmatize(word, 'v') # n = noun, v =verb, a = adjective, r = adverbs, s = satellite adjectives
    print(lemma)

