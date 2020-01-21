
import nltk

paragraph =  """Words aren’t things that computers naturally understand. By 
encoding them in a numeric form, we can apply mathematical rules and do matrix 
operations to them. This makes them amazing in the world of machine learning, 
especially. Take deep learning for example. By encoding words in a numerical 
form, we can take many deep learning architectures and apply them to words. 
Convolutional neural networks have been applied to NLP tasks using word 
embeddings and have set the state-of-the-art performance for many tasks. Even 
better, what we have found is that we can actually pre-train word embeddings 
that are applicable to many tasks. That’s the focus of many of the types we 
will address in this article. So one doesn’t have to learn a new set of 
embeddings per task, per corpora. Instead, we can learn general representation 
which can then be used across tasks."""
               
               
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()