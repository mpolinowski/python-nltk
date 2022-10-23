from nltk.sentiment import SentimentIntensityAnalyzer

# run once
# import nltk
# nltk.download('vader_lexicon')

analyzer = SentimentIntensityAnalyzer()

json_response = { "reviews": [
        { "id": "432rsde34t",
          "title": "Some Title",
          "text": "I regret this purchase... Only buy if you enjoy a keyboard that goes to sleep even when hardwired and takes like 5 seconds to wake up. This means copy and paste will sometimes fail on the copy because keyboard was sleeping or you think you are typing but you are not lol. So if you throw in the fact that you will get random stuck keys and sometimes profile changes what you actually have is just an annoying keyboard. I have never done so many updates on a review, this is what a mistake it is to purchase this hardware. And take it from me, someone who has a lot of razer products and all of which have similar issues... look else where, Razer is not the company you remember, very disappointing.",
          "author": "Some Author"},
        { "id": "67Gfdhnd4",
          "title": "Some Title",
          "text": "I do have a couple of complaints, though. The software is a bit slow and bloated and seems to slow down my startup time on my computer. I am also having issues with the volume wheel - sometimes it will scroll web pages for some reason and does weird things like jump from 50 to 100 or turn up when i turn down. I also wish it was a bit cheaper, but for my requirements, I did not really have much choice.",
          "author": "Some Author"},
        { "id": "sHB8679iasd",
          "title": "Some Title",
          "text": "I purchased the full length Halo Infinite edition with green switches for the tactile clickiness. Needed a second keyboard for work, thought why not try a smaller form factor from the same product line with the same green switches. I am guessing it is from the phantom keycaps, but the green switches are somewhat muffled and mushy in this form factor. I am 50/50 on the phantom edition keycaps for this form factor. I am new to it, so where my muscle memory puts the keys is incorrect. And without static backlighting set, I feel lost while trying to blaze through a long work email. I decided to use my full length version for work and this for my personal gaming desktop since I mainly use a controller or a Logitech G13 when controllers are not supported.",
          "author": "Some Author"},
        { "id": "dhg456wASF",
          "title": "Some Title",
          "text": "I was REALLY enjoying this keyboard (and my wife was enjoying hers too) - however, both keyboard developped an extra keystroke when typing. This lead to extra letters being inputted, which made the keyboard unuseable. Its a shame, because for the price it was a great keyboard. It also fit really well into our Razer ecosystem AND had amazing battery life with the RGB on. Alas, the keystroke issue is too big a hassle to try again, sadly.",
          "author": "Some Author"},
        { "id": "gfdh346Nr",
          "title": "Some Title",
          "text": "This keyboard is excellent, but only if you mod this keyboard, like add some foam, lube the switches, add some painters tape, bandaid mod the stabs and lube them, then switch for some Durocks (cus razer is trying to be cool again and make their own, which quite frankly it's the same as plate mount stabs unlike the huntsman mini) so for the price of this keyboard it is good when amazon discount it but in general for 200$ Nah pass (but I got a deal for only 126$). I kind of get where they use premium metal, good lithium battery, two signal receivers, three battery cutoff the PCB, the power monitor and distribute board with, and the battery integrated itself but still, only if people knew this. They might appreciate it a little more. Still, the transparent bottom plastic with the glowing razer logo scratches way too quickly; that is all I have to say about this keyboard.",
          "author": "Some Author"}
    ]
}

polarity_scores = []

# extract article abstracts and combine them
for review in json_response['reviews']:
    score = analyzer.polarity_scores(review['text'])
    polarity_scores.append(score)

# print(polarity_scores)
# [{'neg': 0.149, 'neu': 0.823, 'pos': 0.027, 'compound': -0.953}, {'neg': 0.042, 'neu': 0.91, 'pos': 0.047, 'compound': 0.1027}, {'neg': 0.042, 'neu': 0.958, 'pos': 0.0, 'compound': -0.659}, {'neg': 0.085, 'neu': 0.683, 'pos': 0.231, 'compound': 0.9397}, {'neg': 0.053, 'neu': 0.822, 'pos': 0.125, 'compound': 0.9109}]

# neg = negativity score
# neu = neutrality score
# pos = positivity score
# compound = sentiment score can range from -1 to 1. closer to 1 = more positive

sentiment_sum = 0

for compound_sentiment in polarity_scores:
    # print(compound_sentiment['compound'])
    sentiment_sum = sentiment_sum + float(compound_sentiment['compound'])

average_sentiment = sentiment_sum / len(polarity_scores)

# print(average_sentiment)
# # 0.06825999999999999

if average_sentiment > 1.3:
    print('INFO :: The average sentiment is POSITIVE')
elif average_sentiment < -0.3:
    print('INFO :: The average sentiment is NEGATIVE')
else:
    print('INFO :: The average sentiment is NEUTRAL')


# INFO :: The average sentiment is NEUTRAL