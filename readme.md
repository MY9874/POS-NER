# This is an NLP project of POS Tagging and Named Entity Recognition

In this project I implemented 2 different algorithms for POS Tagging and 2 different methods for NER:
- greedy_viterbi.py is the implementation of greedy algorithm and viterbi algorthm based on transition and emission probabilities, to solve POS Tagging task, got accuracy of 93.53% and 94.81% respectively


- lstm.py is the implementation of RNN using LSTM cell, used embedding layer for word embedding to solve NER task, got f1 score of 84.76%


- GloVe.py is the implementation of RNN using LSTM cell, but used GloVe vectors for word embedding to solve NER task, got f1 score of 88.72%


- Developed on Python 3.8
- Used PyTorch framework
- testtest