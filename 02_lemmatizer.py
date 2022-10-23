import nltk

# only run once
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

text = 'Experiments on mice at Boston University have spotlighted an ambiguous U.S. policy for research on potentially dangerous pathogens.'
# break text up into single words toLowerCase
tokens = nltk.word_tokenize(text.lower())
# get the pos tag for each token (check if it is verb, noun, etc.)
tags = nltk.pos_tag(tokens)

# print(tags)
# [('experiments', 'NNS'), ('on', 'IN'), ('mice', 'NNS'), ('at', 'IN'), ('boston', 'NN'), ('university', 'NN'), ('have', 'VBP'), ('spotlighted', 'VBN'), ('an', 'DT'), ('ambiguous', 'JJ'), ('u.s.', 'NN'), ('policy', 'NN'), ('for', 'IN'), ('research', 'NN'), ('on', 'IN'), ('potentially', 'RB'), ('dangerous', 'JJ'), ('pathogens', 'NNS'), ('.', '.')]

lemmatizer = nltk.stem.WordNetLemmatizer()
text_lemmas = []

for token, tag in zip (tokens, tags):
    # extract part-of-speach tag
    tag_pos = tag[1][0].lower()
    # print(token, tag_pos)
    # exclude prepositions, articles, etc
    if tag_pos in ['n', 'v', 'a', 'r']:
        lemma = lemmatizer.lemmatize(token, tag_pos)
        text_lemmas.append(lemma)


print(text_lemmas)
# ['experiment', 'mouse', 'boston', 'university', 'have', 'spotlight', 'u.s.', 'policy', 'research', 'potentially', 'pathogen']