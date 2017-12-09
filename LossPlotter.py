import json as json

import matplotlib.pyplot as plt
import numpy as np

def plotLoss(filename):
    jsonFile = open(filename, 'r')
    lossJSON = json.load(jsonFile)
    loss = np.asarray(lossJSON['loss'])
    plt.plot (loss)
    plt.show()

plotLoss('LOSS_10_12_2017_00_31_29.json')