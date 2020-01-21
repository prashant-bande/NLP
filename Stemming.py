
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

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
               
               
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   