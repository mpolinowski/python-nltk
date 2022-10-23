import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

lemmatizer = nltk.stem.WordNetLemmatizer()

# only run once
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

json_response = { "articles": [
        { "id": "432rsde34t",
          "title": "Some Title",
          "abstract": "Experiments on mice at Boston University have spotlighted an ambiguous U.S. policy for research on potentially dangerous pathogens.",
          "author": "Some Author"},
        { "id": "67Gfdhnd4",
          "title": "Some Title",
          "abstract": "The move puts President Biden’s debt relief plan on hold. The court granted a stay in response to an appeal filed by six Republican-led states.",
          "author": "Some Author"},
        { "id": "sHB8679iasd",
          "title": "Some Title",
          "abstract": "The new Communist Party elite will limit potential resistance to Mr. Xi’s agenda of bolstering security and expanding state sway over the economy.",
          "author": "Some Author"},
        { "id": "dhg456wASF",
          "title": "Some Title",
          "abstract": "When Laurene Powell Jobs unveiled a website dedicated to her husband, many wondered if it could change how influential people burnish their legacies.",
          "author": "Some Author"},
        { "id": "gfdh346Nr",
          "title": "Some Title",
          "abstract": "If former President Trump turns down the drama of testifying, his legal team could mount several constitutional and procedural arguments in court.",
          "author": "Some Author"}
    ]
}

# print(json_response['articles'][0]['abstract'])

text = ""

# extract article abstracts and combine them
for article in json_response['articles']:
    # print(article['abstract'])
    text = text + article['abstract'] + ' '

# print(text)


# compare articles to the following search query
query = 'University Boston Experiment'

# get list of single sentences out of combined text
sentences = nltk.sent_tokenize(text)
# print(sentences)
# append query sentence to list
sentences.append(query)


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

# get lemmas out of list of sentences
tv = TfidfVectorizer(tokenizer=get_lemmas)
# generate matrix with weights for each lemma in the given text (how often do they appear)
tf = tv.fit_transform(sentences)

# import pandas as pd
# df = pd.DataFrame(tf.toarray(), columns=tv.get_feature_names_out())
# print(df)
# this returns the matrix of words and their relative weight.
# each row represents a sentence that we fed into the function.
# the last row tf[-1] is the query string
# #      agenda    appeal  argument   bolster   boston   burnish  ...  university    unveil   website    wonder        xi         ’
# # 0  0.000000  0.000000  0.000000  0.000000  0.26162  0.000000  ...     0.26162  0.000000  0.000000  0.000000  0.000000  0.000000
# # 1  0.000000  0.000000  0.000000  0.000000  0.00000  0.000000  ...     0.00000  0.000000  0.000000  0.000000  0.000000  0.305598
# # 2  0.000000  0.395963  0.000000  0.000000  0.00000  0.000000  ...     0.00000  0.000000  0.000000  0.000000  0.000000  0.000000
# # 3  0.263724  0.000000  0.000000  0.263724  0.00000  0.000000  ...     0.00000  0.000000  0.000000  0.000000  0.263724  0.218913
# # 4  0.000000  0.000000  0.000000  0.000000  0.00000  0.288675  ...     0.00000  0.288675  0.288675  0.288675  0.000000  0.000000
# # 5  0.000000  0.000000  0.326545  0.000000  0.00000  0.000000  ...     0.00000  0.000000  0.000000  0.000000  0.000000  0.000000
# # 6  0.000000  0.000000  0.000000  0.000000  0.57735  0.000000  ...     0.57735  0.000000  0.000000  0.000000  0.000000  0.000000

# # [7 rows x 59 columns]

# Now we can calculate the relative similarity
# of each sentence to the query string

coefficients = cosine_similarity(tf[-1], tf)

# print(coefficients)
# the result is that the query string matches itself by 100%
# and the next best match is the first sentence
# [[0.4531384 0.        0.        0.        0.        0.        1.]]

# now we can sort that list and extract the matching sentence
# index = coefficients.argsort()[0]
# the result is a nested list use zero index or flatten() to extract
# print(index)

# 6 represents the query string and 0 is the position of the best match
# [1 2 3 4 5 0 6]

# so we need to extract the second to last
index = coefficients.argsort().flatten()[-2]

# we can use the index to get the sentence with the best match
print(sentences[index])

# Result:
# Experiments on mice at Boston University have spotlighted an ambiguous U.S. policy for research on potentially dangerous pathogens.