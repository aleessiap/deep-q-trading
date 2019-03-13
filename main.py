from DeepQTrading import DeepQTrading
import datetime
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import sys


nb_actions = 3   #long, short, hold 

#rete neurale
model = Sequential()
model.add(Flatten(input_shape=(50,1,68)))
model.add(Dense(256,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(512,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(256,activation='linear'))
model.add(LeakyReLU(alpha=.001)) 
model.add(Dense(nb_actions))
model.add(Activation('linear'))




#inizializzazione agente
dqt = DeepQTrading(
    model=model,
    explorations=[(0.1,100)],                   #casualità , num volte che il dataset viene passato 
    trainSize=datetime.timedelta(days=365),     #giorni training
    validationSize=datetime.timedelta(days=30), #giorni validation
    testSize=datetime.timedelta(days=30),       #giorni test
    outputFile="output.csv",                    #file in cui scrive i risultati
    begin=datetime.datetime(2004,1,1,0,0,0,0),  #inizio analisi
    end=datetime.datetime(2017,12,1,0,0,0,0),   #fine analisi
    nbActions=nb_actions                        #num azioni possibili
    )

dqt.run()                                       #avvio dell'agente
dqt.end()                                       #end agente