
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
               

from gensim.models import Word2Vec
from nltk.corpus import stopwords
               
# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)


words = model.wv.vocab

# Finding Word Vectors
vector = model.wv['deep']

# Most similar words
similar = model.wv.most_similar('learn')