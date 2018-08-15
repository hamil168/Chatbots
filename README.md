# Chatbots

_Exploring Chatbots_

This repository ~~may~~ will be a bit messy as it develops.

Notebooks created with https://colab.research.google.com
Training performed on Colab, locally on my GTX 970 or (soon!) JBCurtain's JP Machine


The goal of this project is three-fold:

1) Build a chatbot on a Seq2Seq LSTM using TensorFlow and a publically available data set **_(CURRENT)_**
1) Explore ways to adapt existing chatbots to manifest the speech patterns for speakers given limited data sets
1) Create a chatbot that talks like a specific character archetype using text collected from public domain books or other sources

## Part 1: Seq2Seq LSTM Chatbot
- **Data:** Conversations Cornell Movie Dialogs Corpus (CMD)
- https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
- Based on a tutorial from www.superdatascience.com (I recommend their podcast)
  - also a number of github repos, many of which were based on other, similar Resources
  - also notes from Ng's Deep Learning specialization on Coursera
  - all of those chatbots were done on TF 1.0 through 1.4 at the newest
  - this chatbot is in TF 1.9/1.10, which had some nice new wrappers

The CMD data has >220,000 exchanges between >10,000 pairs of characters and 617 movies.

This data will be cleaned and transformed into 'call-response' pairs that will serve as training inputs and outputs in the model.
- Assume a conversation between A and B has sets of phrases Ax and Bx from the movie transcriptions.
  - A conversation phrase sequence abbreviated as A1 B1 A2 B2 A3 A4 B3 will be transformed to (A1 B1) (B1 A2) (A2 B2) (B2 A3) (A4 B3)
- Speaker and responder identities are not preserved.

***(in development!)***

**Pt. 1 Status**
- Architecture uses 1-way LSTM with Bahdanau attention
  - 25 word inputs and outputs
  - encoders trained as part of the model
- TF 1.9/1.10 architecture has been debugged. The algorithms officially train the model
  - Using small hyperparameters and partial training data for agility (in process)
    - oom error | answer lengths not truncated (fixed)
    - increased training set, rnn size, layers, batch size
  - Has not been error-analyzed and validated yet
    - WIP: training larger models
- To-Be-Deployed for testing.
- TODO:
  - Implement save trained models (done)
  - Implement load from saved (done)
  - Implement timer
  - Render deployable for testing, test
    - hacked solution done, needs additional work
  - Do an initial validation of model before moving to more complicated models
- SOMEDAYs:
  - Explore LSTM hyperparamter Search
  - Regularization (currently dropout 0.5)
  - Explore using BiLSTMs
  - Explore using BeamSearch
  - Explore other encoders, etc.
  - Explore other preprocessing
  - Learn more about buckets (done the right way)
  - queues and input optimizations

  ### References:
  __Blogs__
  - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
  - http://suriyadeepan.github.io/2016-06-28-easy-seq2seq/
    - many good links
  - http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/
  - https://github.com/DongjunLee/conversation-tensorflow
  - https://github.com/hb-research/notes/blob/master/notes/neural_text_generation.md
  - https://adeshpande3.github.io/How-I-Used-Deep-Learning-to-Train-a-Chatbot-to-Talk-Like-Me
  - https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077
    - includes links to many papers
  - https://towardsdatascience.com/personality-for-your-chatbot-with-recurrent-neural-networks-2038f7f34636
  __Papers__
  - "Neural Conversation Models" https://arxiv.org/abs/1506.05869
  __TensorFlow__
  - https://www.tensorflow.org/tutorials/
  - https://github.com/chiphuyen/stanford-tensorflow-tutorials
  - https://github.com/tensorflow/nmt
