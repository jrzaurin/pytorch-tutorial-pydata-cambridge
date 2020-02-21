#!/usr/bin/env bash

# Download Adult
ADULT=./data
mkdir -p $ADULT
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O $ADULT/adult_train.csv
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test -O $ADULT/adult_test.csv
cd $ADULT
sed -i '' 1d adult_test.csv
cd ..