import numpy as np
from keras.models import load_model
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_iter', type=int,default=0, help='Starting iteration number')
parser.add_argument('--total_iter', type=int,default=15, help='Number of iterations to calculate')
parser.add_argument('--fake_data', type=int,default=-1, help='Fake data number')
parser.add_argument('--trials', type=int,default=15, help='Number of trials')
parser.add_argument('--throws', type=int,default=100, help='Number of throws')
parser.add_argument('--save_name', type=str,default="Test", help='Save file name for result')
parser.add_argument('--weight_prefix', type=str,default="t2k_FDS0_MCStat", help='Prefix for weights files')
parser.add_argument('--weight_dir', type=str,default="weights_omnifold/", help='Directory for weights files')


flags = parser.parse_args()

dataDir = './'
weights_dir = flags.weight_dir

fakeIdx = ""
if flags.fake_data != -1:
    fakeIdx = str(flags.fake_data)

saveName = flags.save_name
print(saveName)

inputVars = 11
activationFunc = 'leaky_relu'
inputs2 = Input((inputVars, ))
layer2 = Dense(100,activation=activationFunc)(inputs2)
layer2 = Dense(100,activation=activationFunc)(layer2)
layer2 = Dense(100,activation=activationFunc)(layer2)
layer2 = Dense(100,activation=activationFunc)(layer2)
outputs2 = Dense(1,activation='sigmoid')(layer2)
model2 = Model(inputs=inputs2, outputs=outputs2)

totalTrials = int(flags.trials) #20
totalThrows = int(flags.throws) #100

for finalIter in range(flags.start_iter, flags.start_iter + flags.total_iter):
    allRaw = []
    allOmni = []
    allTrials = []
    for throwIdx in range(totalThrows):
        prefix = "%s%d" % (flags.weight_prefix, throwIdx)

        mc_truth = np.load(dataDir+'mc_vals_truth_Topology.npy')

        omniweights = [0 for val in mc_truth]
        rawweights = [0 for val in mc_truth]
        allweights = []
        for trial in range(totalTrials):
            print(trial)
            model2.load_weights(weights_dir+'%s_trial%d_iter%d_step2.h5' % (prefix,trial,finalIter))
            thisWeights = model2.predict(mc_truth, batch_size=4096, verbose=0)[:,0]
            rawweights += thisWeights
            omniweights += thisWeights / (1. - thisWeights)
            allweights.append(thisWeights / (1. - thisWeights))
        if totalTrials > 0:
            omniweights /= totalTrials
            rawweights /= totalTrials
            rawweights = rawweights / (1.- rawweights)
        else:
            omniweights = 1
            rawweights = 1
        allOmni.append(omniweights)
        allRaw.append(rawweights)
        allTrials.append(allweights)


    np.save("%s/OmnifoldNNAverage_%s_Iter%d.npy" % (weights_dir, saveName, finalIter), np.array(allRaw, dtype=np.single))
    np.save("%s/OmnifoldAverage_%s_Iter%d.npy" % (weights_dir, saveName, finalIter), np.array(allOmni, dtype=np.single))
    np.save("%s/OmnifoldAllReweights_%s_Iter%d.npy" % (weights_dir, saveName, finalIter), np.array(allTrials, dtype=np.single))
