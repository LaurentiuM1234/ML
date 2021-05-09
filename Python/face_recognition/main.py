from ImageHandler import ImageHandler
from PCHandler import PCHandler
from Predictor import Predictor
import numpy as np


train_path = '/Users/laurentiumihalcea/Desktop/MLStuff/Data/train_samples'
test_path = '/Users/laurentiumihalcea/Desktop/MLStuff/Data/test_samples'

a = Predictor(train_path, 10)
a.batch_prediction(test_path)