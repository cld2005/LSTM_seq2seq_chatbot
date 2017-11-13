#!/usr/bin/env bash
scp -i test1.pem lstm_seq2seq_conv.py ubuntu@ec2-54-173-151-242.compute-1.amazonaws.com:~/seq2seq_keras/
scp -i test1.pem lstm_seq2seq.py ubuntu@ec2-54-173-151-242.compute-1.amazonaws.com:~/seq2seq_keras/
scp -i test1.pem conv/south_park.txt ubuntu@ec2-54-173-151-242.compute-1.amazonaws.com:~/seq2seq_keras/conv
