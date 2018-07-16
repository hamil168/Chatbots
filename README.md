# Chatbots

_Exploring Chatbots_

This repository may be a bit messy as it develops.

Notebooks created with https://colab.research.google.com

The goal of this project is three-fold:

1) Build a chatbot on a Seq2Seq LSTM using TensorFlow and a publically available data set **_(CURRENT)_**
1) Explore ways to adapt existing chatbots to manifest the speech patterns for speakers given limited data sets
1) Create a chatbot that talks like a specific character archetype using text collected from public domain books or other sources

## Part 1: Seq2Seq LSTM Chatbot
- **Data:** Conversations Cornell Movie Dialogs Corpus (CMD)
- https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
- Based on a tutorial from www.superdatascience.com (I also recommend their podcast)

The CMD data has >220,000 exchanges between >10,000 pairs of characters and 617 movies. 

This data will be cleaned and transformed into 'call-response' pairs that will serve as training inputs and outputs in the model.
- Assume a conversation between A and B has sets of phrases Ax and Bx from the movie transcriptions.
  - A conversation phrase sequence abbreviated as A1 B1 A2 B2 A3 A4 B3 will be transformed to (A1 B1) (B1 A2) (A2 B2) (B2 A3) (A4 B3)
- Speaker and responder identities are not preserved.

(in development)
