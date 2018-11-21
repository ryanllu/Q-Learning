import numpy as np


class QTable:
    def __init__(self,shape):
        self.table=np.zeros(np.prod(shape)).reshape(*shape)

    def getQ(self,state,action):
        target=self.table.copy()
        for component in state:
            target=target[component]
        return target[action]
    
    def setQ(self,new_value,state,action):
        indexes=""
        for component in state:
            indexes+="["
            indexes+=str(component)
            indexes+="]"
        indexes+="["
        indexes+=str(action)
        indexes+="]"
        exec("self.table"+indexes+"=new_value")
    
    def getMaxQ(self,state):
        target=self.table.copy()
        for component in state:
            target=target[component]
        return np.max(target)

    def getBestAction(self, state):
        target=self.table.copy()
        for component in state:
            target=target[component]
        return np.where(target==np.max(target))[0][0]
        

class QLearning:

    def __init__(
                 self,
                 environment,
                 state_partition=5,
                 learning_rate=0.5,
                 explore_rate=0.5,
                 discount_rate=0.5):
        
        observation_dict=environment.observation_space.__dict__
            
        self.environment=environment

        self.max_observation=observation_dict["high"]
        self.min_observation=observation_dict["low"]
        self.observation_range=self.max_observation-self.min_observation
        self.observation_dimension=len(self.observation_range)

        if(type(state_partition)==int):
            self.state_partition=[state_partition for i in self.observation_range]
        elif(type(state_partition)==list):
            self.state_partition=state_partition
        
        self.qTable=QTable(self.state_partition+[self.environment.action_space.n])
                           
        self.learning_rate=learning_rate
        self.explore_rate=explore_rate
        self.discount_rate=discount_rate

        self.current_observation=None
        self.previous_observation=None
        self.last_action=None

        np.random.seed(999)

    def getAction(self,state):
        threshold=np.random.rand()
        if(threshold<=self.explore_rate):
            action=self.environment.action_space.sample()
        else:
            action=self.qTable.getBestAction(state)
        return action
            
    def getReward(self,lambda_function):
        return lambda_function
    
    def getState(self,observation):
        state=[]

        for index,value in enumerate(observation):
            step=self.observation_range[index]/self.state_partition[index]
            threshold=self.min_observation[index]+step
            state_code=0
            for _ in range(1,self.state_partition[index]):
                if(value<=threshold):
                    break
                else:
                    state_code+=1
                    threshold+=step
            state.append(state_code)
        return state
    
    def updateQTable(self,reward):
        state=self.getState(self.previous_observation)
        next_state=self.getState(self.current_observation)
        current_q=self.qTable.getQ(state,self.last_action)
        max_future_q=self.qTable.getMaxQ(next_state)
        self.qTable.setQ(self.learning_rate*((-1*(current_q))+(self.discount_rate*max_future_q)+reward),state,self.last_action)

    def train_render(self,episodes,max_steps):
        for episode in range(episodes):
            self.current_observation=self.environment.reset()
            rewards=200
            for step in range(max_steps):
                state=self.getState(self.current_observation)
                self.last_action=self.getAction(state)
                next_obs, reward, done, info = self.environment.step(self.last_action)
                rewards+=reward
                self.previous_observation=self.current_observation
                self.current_observation=next_obs
                self.environment.render()
                if(done):
                    break
                else:
                    self.updateQTable(rewards)
                    obs=next_obs
            if(step<199):
                result="Successful"
                print("Episode {} : {}".format(episode,result))
            else:
                result="Unsuccessful"
                print("Episode {} : {}".format(episode,result))
            
    def train(self,episodes,max_steps):
        for episode in range(episodes):
            self.current_observation=self.environment.reset()
            rewards=200
            for step in range(max_steps):
                state=self.getState(self.current_observation)
                self.last_action=self.getAction(state)
                next_obs, reward, done, info = self.environment.step(self.last_action)
                rewards+=reward
                self.previous_observation=self.current_observation
                self.current_observation=next_obs
                if(done):
                    break
                else:
                    self.updateQTable(rewards)
                    obs=next_obs
            if(step<199):
                result="Successful"
                print("Episode {} : {}".format(episode,result))
            else:
                result="Unsuccessful"
                print("Episode {} : {}".format(episode,result))
        
