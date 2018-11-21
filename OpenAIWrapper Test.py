from OpenAIWrapper import QLearning
from OpenAIWrapper import QTable
import gym

env=gym.make("MountainCar-v0")

model=QLearning(environment=env,state_partition=[10,10],learning_rate=0.05,explore_rate=0.3,discount_rate=0.1)

model.train(1000,200)


