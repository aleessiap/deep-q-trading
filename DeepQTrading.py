from SpEnv import SpEnv
from Callback import ValidationCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
import pandas as pd
import datetime

class DeepQTrading:
    def __init__(self, model, explorations, trainSize, validationSize, testSize, outputFile, begin, end, nbActions, operationCost=0):
        self.policy = EpsGreedyQPolicy()
        self.memory = SequentialMemory(limit=10000, window_length=50)
        
        self.model=model
        self.explorations=explorations
        self.trainSize=trainSize
        self.validationSize=validationSize
        self.testSize=testSize
        self.currentStartingPoint = begin
        self.endingPoint=end
        self.nbActions=nbActions
        self.operationCost = operationCost

        self.walkSize=trainSize+validationSize+testSize
        self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=400, 
        target_model_update=1e-1, enable_double_dqn=True, enable_dueling_network=True)
        self.agent.compile(Adam(lr=1e-3), metrics=['mae'])
        self.agent.save_weights("q.weights", overwrite=True)

        #pandas per csv
        self.sp = pd.read_csv('./dataset/sp500Hour.csv')
        self.sp['Datetime'] = pd.to_datetime(self.sp['Date'] + ' ' + self.sp['Time'])
        self.sp = self.sp.set_index('Datetime')
        self.sp = self.sp.drop(['Date','Time'], axis=1)
        self.sp = self.sp.index

        self.trainer=ValidationCallback()
        self.validator=ValidationCallback()
        self.tester=ValidationCallback()

        self.outputFile=open(outputFile, "w+")
        self.outputFile.write("date,trainAccuracy,trainCoverage,trainReward,validationAccuracy,validationCoverage,validationReward,testAccuracy,testCoverage,testReward\n")


    def run(self):
        env=" "

        iteration=0
        """
        minLimit=None
        while(minLimit is None):
            try:
                minLimit = self.sp.get_loc(self.currentStartingPoint)
            except:
                self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)
        maxLimit=None
        while(maxLimit is None):
            try:
                maxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize)
            except:
                self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

        eps=self.explorations[0]
        self.policy.eps = eps[0]
        print(eps[1])
        env = SpEnv(operationCost=self.operationCost,minLimit=minLimit,maxLimit=maxLimit)
        for i in range(0,eps[1]):
            print(str(i) + " " + str(self.trainSize.days-10))
            self.agent.fit(env,nb_steps=self.trainSize.days-65,visualize=False,callbacks=[self.trainer],verbose=0)#problema con nb_steps (devo cercare di farlo in episodi)
            env.resetEnv()

        (episodes,trainCoverage,trainAccuracy,trainReward)=self.trainer.getInfo()
        print(str(iteration) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward))
        self.trainer.reset()
        """

        while(self.currentStartingPoint+self.walkSize <= self.endingPoint):     #walk
            iteration+=1
            
            #reset 
            del(self.memory)
            del(self.agent)
            self.memory = SequentialMemory(limit=10000, window_length=50)
            self.agent = DQNAgent(model=self.model, policy=self.policy,  nb_actions=self.nbActions, memory=self.memory, nb_steps_warmup=400, 
                                target_model_update=1e-1, enable_double_dqn=True, enable_dueling_network=True)
            self.agent.compile(Adam(lr=1e-3), metrics=['mae'])
            self.agent.load_weights("q.weights")




#training
            minLimit=None
            while(minLimit is None):
                try:
                    minLimit = self.sp.get_loc(self.currentStartingPoint)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            maxLimit=None
            while(maxLimit is None):
                try:
                    maxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)


            date=self.currentStartingPoint
            for eps in self.explorations:
                self.policy.eps = eps[0] #casualità
                del(env)
                env = SpEnv(operationCost=self.operationCost,minLimit=minLimit,maxLimit=maxLimit) #ambiente

                #dati all'agente
                for _ in range(0,eps[1]): 
                    self.trainer.reset()
                    self.agent.fit(env, nb_steps=self.trainSize.days-65, visualize=False, callbacks=[self.trainer], verbose=0) 
                    env.resetEnv()
                    
            (_,trainCoverage,trainAccuracy,trainReward)=self.trainer.getInfo()
            print(str(iteration) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward))

#validation
            minLimit=maxLimit
            maxLimit=None
            while(maxLimit is None):
                try:
                    maxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)
            del(env)
            env=SpEnv(operationCost=self.operationCost, minLimit=minLimit, maxLimit=maxLimit)

            self.agent.test(env, nb_episodes=self.validationSize.days-10, visualize=False, callbacks=[self.validator], verbose=0)
            (_,validCoverage, validAccuracy, validReward)=self.validator.getInfo()
            print(str(iteration) + " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))

            self.validator.reset()

            # minLimit=maxLimit
            # minLimit=None
            # while(minLimit is None):
            #     try:
            #         minLimit = self.sp.get_loc(self.currentStartingPoint+self.walkSize)
            #     except:
            #         self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)
            #outputFile.write(str(iteration) + "," + str(trainAccuracy)+ "," + str(trainCoverage)+ "," + str(trainReward)+ "," + str(validAccuracy)+ "," + str(validCoverage)+ "," + str(validReward) + "\n")
            #print(str(iteration) + " TRAIN:  acc: " + str(trainAccuracy)+ " cov: " + str(trainCoverage)+ " rew: " + str(trainReward)+ " VALID:  acc: " + str(validAccuracy)+ " cov: " + str(validCoverage)+ " rew: " + str(validReward))
            
#testing
            minLimit=maxLimit
            maxLimit=None
            while(maxLimit is None):
                try:
                    maxLimit = self.sp.get_loc(self.currentStartingPoint+self.trainSize+self.validationSize+self.testSize)
                except:
                    self.currentStartingPoint+=datetime.timedelta(0,0,0,0,0,1,0)

            del(env)
            env=SpEnv(operationCost=self.operationCost,minLimit=minLimit,maxLimit=maxLimit)

            self.agent.test(env,nb_episodes=self.validationSize.days-10,visualize=False,callbacks=[self.tester],verbose=0)
            (_,testCoverage,testAccuracy,testReward)=self.tester.getInfo()
            print(str(iteration) + " TEST:  acc: " + str(testAccuracy)+ " cov: " + str(testCoverage)+ " rew: " + str(testReward))

            self.tester.reset()

            print(" ")
#scrittura risultati            
            self.outputFile.write(str(date)+","+str(trainAccuracy)+","+str(trainCoverage)+","+str(trainReward)+","+str(validAccuracy)+","+str(validCoverage)+","+str(validReward)+","+str(testAccuracy)+","+str(testCoverage)+","+str(testReward)+"\n")
        
#aggiorna punto di inizio, fa un passo        
            self.currentStartingPoint+=self.testSize


    def end(self):
        import os 
        self.outputFile.close()
        os.remove("q.weights")

