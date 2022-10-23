import nltk

# only run once
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

text1 = 'The Experiments on mice...'
text2 = 'The Experiment on a mouse...'

lemmatizer = nltk.stem.WordNetLemmatizer()

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

    # print(text_lemmas)
    return text_lemmas

source1 = get_lemmas(text1)
source2 = get_lemmas(text2)


print(source1 == source2)
# True