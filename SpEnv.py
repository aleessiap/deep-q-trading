import gym
from gym import spaces
import numpy
import pandas
from datetime import datetime
from MergedDataStructure import MergedDataStructure

class SpEnv(gym.Env):
    
    continuous = False

    def __init__(self, minLimit=None, maxLimit=None, operationCost = 0, observationWindow = 40, outputFile = ""):
        self.episodio=1

#pandas
        spTimeserie = pandas.read_csv('./dataset/sp500Hour.csv')[minLimit:maxLimit] # opening the dataset
        Date = spTimeserie.ix[:, 'Date'].tolist()
        Time = spTimeserie.ix[:, 'Time'].tolist()
        Open = spTimeserie.ix[:, 'Open'].tolist()       #valore ad apertura
        High = spTimeserie.ix[:, 'High'].tolist()       #picco più alto valore
        Low = spTimeserie.ix[:, 'Low'].tolist()         #picco più basso valore
        Close = spTimeserie.ix[:, 'Close'].tolist()     #valore a chiusura
        Volume = spTimeserie.ix[:, 'Volume'].tolist()   #num volte in cui è stata fatta un'azione nel periodo di tempo

        self.weekData = MergedDataStructure(delta=8,filename="./dataset/sp500Week.csv")
        self.dayData = MergedDataStructure(delta=20,filename="./dataset/sp500Day.csv")
        
        self.output=False

        if(outputFile!=""): # Managing file output
            self.output=True
            self.outputFile=open(outputFile, "w+")
            self.outputFile.write("date,open,close,reward,possible_gain,hit\n")

        self.low = numpy.array([-numpy.inf])     #-infinito
        self.high = numpy.array([+numpy.inf])    #+infinito
        self.action_space = spaces.Discrete(3)   # the action space is just 0,1,2 which means nop,buy,sell
       
       
        self.observation_space = spaces.Box(self.low, self.high, dtype=numpy.float32)


        self.history=[]
        self.observationWindow = observationWindow
        self.currentObservation = observationWindow
        
        #print(self.currentObservation)
        self.operationCost=operationCost
        self.done = False
        self.limit = len(Open)

        for i in range(0,self.limit): # organizing the dataset as a list of dictionaries
            self.history.append({'Date' : Date[i],'Time' : Time[i], 'Open': Open[i], 'High': High[i], 'Low': Low[i], 'Close': Close[i], 'Volume': Volume[i] })
        
        self.nextObservation=0
        # print(self.currentObservation)
        while(self.history[self.currentObservation]['Date']==self.history[(self.currentObservation+self.nextObservation)%self.limit]['Date']):
            self.nextObservation+=1
        # print(self.limit)
        self.reward = None
        self.possibleGain = 0
        self.openValue = 0
        self.closeValue = 0


    def step(self, action):
        self.reward=0
        weekList = []
        dayList = []

        dayList=self.dayData.get(self.history[self.currentObservation]['Date'])
        weekList=self.weekData.get(self.history[self.currentObservation]['Date'])
        
        currentData = self.history[self.currentObservation-self.observationWindow:self.currentObservation] 

        currentData=currentData + dayList + weekList

        closeMinusOpen=list(map(lambda x: (x["Close"]-x["Open"])/x["Open"],currentData)) #reward
        # high=list(map(lambda x: x["High"],currentData))
        # low=list(map(lambda x: x["Low"],currentData))
        # volume=list(map(lambda x: x["Volume"],currentData))

        self.nextObservation=0
        while(self.history[self.currentObservation]['Date']==self.history[(self.currentObservation+self.nextObservation)%self.limit]['Date']):
            self.closeValue=self.history[(self.currentObservation+self.nextObservation)%self.limit]['Close']
            self.nextObservation+=1

        self.openValue = self.history[self.currentObservation]['Open']
        self.possibleGain = (self.closeValue - self.openValue)*50
        if(action == 1): #long
            self.reward = self.possibleGain-self.operationCost
        elif(action==2): #short
            self.reward = (-self.possibleGain)-self.operationCost
        else:   #hold
            self.reward = 0


        #self.currentObservation+=self.nextObservation

        self.done=True
        

        state = numpy.array([closeMinusOpen])
        #state = numpy.array([closeMinusOpen,high,low,volume])
        #print(str(action) + " " + str(self.reward))
        
        return state, self.reward, self.done, {}
        

    def reset(self):
        self.done = False
        self.episodio+=1
        self.nextObservation=0
        while(self.history[self.currentObservation]['Date']==self.history[(self.currentObservation+self.nextObservation)%self.limit]['Date']):
            self.nextObservation+=1
            if((self.currentObservation+self.nextObservation)>=self.limit):
                print("Balordo: episodio " + str(self.episodio) )
            
        #print(self.limit)
        self.reward = None
        self.possibleGain = 0
        self.openValue = 0
        self.closeValue = 0

        if(self.output and self.reward!=None):
            self.outputFile.write(
                str(self.history[self.currentObservation]['Date']) + "," + 
                str(self.openValue) + "," + 
                str(self.closeValue) + "," + 
                str(self.reward) + "," + 
                str(self.possibleGain) + "," + 
                str((1 if (self.reward>=self.possibleGain and self.reward>=0) else 0)) + "\n")
        
        dayList = []
        weekList = []



        dayList=self.dayData.get(self.history[self.currentObservation]['Date'])
        weekList=self.weekData.get(self.history[self.currentObservation]['Date'])
        
        currentData = self.history[self.currentObservation-self.observationWindow:self.currentObservation] 

        currentData=currentData + dayList + weekList

        self.currentObservation+=self.nextObservation
        self.currentObservation%=self.limit

        if(self.currentObservation<self.observationWindow):
            self.currentObservation=self.observationWindow
            self.reset()
        self.nextObservation=0
        closeMinusOpen=list(map(lambda x: (x["Close"]-x["Open"])/x["Open"],currentData))
        #high=list(map(lambda x: x["High"],currentData))
        #low=list(map(lambda x: x["Low"],currentData))
        #volume=list(map(lambda x: x["Volume"],currentData))

        
        state = numpy.array([closeMinusOpen])
        #state = numpy.array([closeMinusOpen,high,low,volume])
        return state

    def resetEnv(self):
        self.currentObservation=self.observationWindow
        self.episodio=1