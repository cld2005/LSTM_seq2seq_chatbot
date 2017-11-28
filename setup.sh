#!/usr/bin/env
git clone https://github.com/cld2005/LSTM_seq2seq.git
mkdir glove.6B
cd glove.6B
wget https://s3.amazonaws.com/cld2005.glove.100/glove.6B.100d.txt
cd ../LSTM_seq2seq