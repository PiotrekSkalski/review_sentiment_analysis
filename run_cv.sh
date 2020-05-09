#!/usr/bin/env bash

echo '-----CROSS-VALIDATION-----' > cv.txt
echo '--- baseline model ---' >> cv.txt
python train_baseline.py -s 30 -cv -k 10
tail -n2 log.txt >> cv.txt

echo '--- GRU with concat pooling head ---' >> cv.txt
python train_GRU_concatpool.py -s 30 -cv -k 10
tail -n2 log.txt >> cv.txt

echo '--- GRU with self attention head ---' >> cv.txt
python train_GRU_selfattention.py -s 30 -cv -k 10
tail -n2 log.txt >> cv.txt

echo '--- CNN model ___' >> cv.txt
python train_CNN.py -s 30 -cv -k 10
tail -n2 log.txt >> cv.txt

