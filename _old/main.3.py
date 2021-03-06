import IntradayPolicy
import SpEnv
from Callback import ValidationCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from datetime import datetime
import sys


#2006-2015   2016
trainEnv = SpEnv.SpEnv(operationCost = 0,minLimit=13378, maxLimit=74336)
validationEnv =SpEnv.SpEnv(operationCost = 0, minLimit=74336, maxLimit=80500)

validator = ValidationCallback()
trainer = ValidationCallback()

nb_actions = trainEnv.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(50,4,68)))
model.add(Dense(512,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(1024,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(512,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy()

memory = SequentialMemory(limit=10000, window_length=50)
dqn = DQNAgent(model=model, policy=policy,  nb_actions=nb_actions, memory=memory, nb_steps_warmup=400,
target_model_update=1e-1, enable_double_dqn=True, enable_dueling_network=True)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

outputFile=open("2105.csv", "w+")
outputFile.write("iteration,trainAccuracy,trainCoverage,trainReward,validationAccuracy,validationCoverage,validationReward\n")
iteration = 0



for i in range(0,100):
    dqn.fit(trainEnv, nb_steps=3000, visualize=False, callbacks=[trainer], verbose=0)
    (episodes,trainCoverage,trainAccuracy,trainReward)=trainer.getInfo()
    dqn.test(validationEnv, nb_episodes=300, verbose=0, callbacks=[validator], visualize=False)
    (episodes,validCoverage,validAccuracy,validReward)=validator.getInfo()
    outputFile.write(str(iteration) + "," + str(trainAccuracy)+ "," + str(trainCoverage)+ "," + str(trainReward)+ "," + str(validAccuracy)+ "," + str(validCoverage)+ "," + str(validReward) + "\n")
    print(str(iteration) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward)+ " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))
    iteration+=1
    validator.reset()
    trainer.reset()


dqn.save_weights(filepath="LongShort.weights",overwrite=True)

outputFile.close()