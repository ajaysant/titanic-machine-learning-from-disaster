## Import the needed referances
import pandas as pd
import numpy as np
import csv as csv

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

## Shuffle the datasets
from sklearn.utils import shuffle

## Learning curve
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

addpoly = True
plot_lc = 0  # 1--display learning curve/ 0 -- don't display

# loading the data sets from the csv files
print('--------load train & test file------')
train_dataset = pd.read_csv(
    '/Users/312076/Desktop/Learning/git_hub/titanic-machine-learning-from-disaster/input/train.csv')
test_dataset = pd.read_csv(
    '/Users/312076/Desktop/Learning/git_hub/titanic-machine-learning-from-disaster/input/test.csv')

print('train dataset: %s, test dataset %s' % (str(train_dataset.shape), str(test_dataset.shape)))
train_dataset.head()

train_dataset.head()

