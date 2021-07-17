import numpy as np
import random
import math
from itertools import groupby
from itertools import product
from itertools import permutations


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        ## Action space 5 locations - (p,q) 5*4 + (0,0)
        self.action_space = [[i,j] for i in range(m) for j in range(m) if i!=j or i==0]
        ## State space 5 locations(X), 24 hours(T), 7 Days(D)
        self.state_space = [(X,T,D) for X in range(m) for T in range(t)  for D in range(d)]
        self.state_init = random.choice(self.state_space)
        self.total_time = 0
        self.max_time = 720
        # Start the first round
        self.reset()
    
    ### Defining basic functions for state comprehension
    def state_get_loc(self,state): 
        return state[0]

    def state_get_time(self,state): 
        return state[1]
        
    def state_get_day(self,state): 
        return state[2]

    ## Encoding state (or state-action) for NN input
    #### Based on 'Deep Reinforcement Learning for List-wise Recommendations' paper section 1.2 -- High  State space small actions space therefore Architecture 2 ( as per problem statement) where input is only state
    def state_encod_arch(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        # state_encod = [0 for _ in range(m+t+d)]
        # state_encod[self.state_get_loc(state)] = 1
        # state_encod[m+self.state_get_time(state)] = 1
        # state_encod[m+t+self.state_get_day(state)] = 1
        # state_encod_np = np.array(state_encod)


        state_encod = np.zeros((m+t+d))
        state_encod[state[0]] = 1
        state_encod[m + np.int(state[1])] = 1
        state_encod[m + t + np.int(state[2])] = 1

        
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

    #     return state_encod

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        else:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        #### add 0 to possible actions index to avoid iteration over empty list in imported module env.py in agent file
        possible_actions_index = random.sample(range(1,m*(m-1)+1), requests) 
        actions = [self.action_space[i] for i in possible_actions_index]

        # [0, 0] is not a 'request', but it is one of the possible actions
        actions.append([0,0])
        possible_actions_index.append(0)
        return possible_actions_index, actions

    
    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        start_loc, time, day = state
        pickup, drop = action
        # 𝑅(𝑠 = 𝑋𝑖𝑇𝑗𝐷𝑘) = 𝑅𝑘 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞)) − 𝐶𝑓 ∗ (𝑇𝑖𝑚𝑒(𝑝, 𝑞) + 𝑇𝑖𝑚𝑒(𝑖, 𝑝)) 𝑎 = (𝑝, 𝑞) , - 𝐶𝑓 𝑎 = (0,0)
        if action == (0,0):
          return -C
        else:
          time_till_pickup = Time_matrix[start_loc,pickup,time,day]
          time_next = int((time + time_till_pickup) % t )
          day_next = int((day + (time + time_till_pickup)//t) % d)
          time_to_drop_from_pickup = Time_matrix[pickup, drop,time_next, day_next]
          revenue = R * time_to_drop_from_pickup
          fuel_cost = C * (time_to_drop_from_pickup + time_till_pickup)
          reward = revenue - fuel_cost
        return reward

#   def time_taken_for_action(self,state,action,Time_matrix):
#         start_loc, time, day = state
#         pickup, drop = action
        
#         if action == (0,0):
#             time_till_next_action = 1
        
#         else:
#             time_till_pickup = np.int(Time_matrix[start_loc,pickup,time,day])
#             time_next = np.int((time + time_till_pickup) % t )
#             day_next = np.int((day + (time + time_till_pickup)//t) % d)
#             time_to_drop_from_pickup = np.int(Time_matrix[pickup, drop, time_next, day_next])
#             time_till_next_action = time_till_pickup + time_to_drop_from_pickup
#         return time_till_next_action

    # def is_terminal(self,time_total):
    #     if time_total>=720:
    #         return True
    #     else: 
    #         return False

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        start_loc, time, day = state
        pickup, drop = action

        if action == [0,0]:
            time_till_next_action = 1
            time_next = np.int((time + time_till_next_action) % t )
            day_next = np.int((day + (time + time_till_next_action)//t) % d)
            next_state = (start_loc,time_next,day_next) 
            self.total_time += time_till_next_action

        else:
            time_till_pickup = Time_matrix[start_loc,pickup,time,day]
            time_next = np.int((time + time_till_pickup) % t )
            day_next = np.int((day + (time + time_till_pickup)//t) % d)
            time_to_drop_from_pickup = np.int(Time_matrix[pickup, drop, time_next, day_next])
            time_till_next_action = time_till_pickup + time_to_drop_from_pickup
            self.total_time += time_till_next_action
            next_state = (drop,time_next,day_next) 

        if self.total_time >= self.max_time: 
            terminal_bool = True
            self.total_time = 0
        else:
            terminal_bool = False 

        return next_state, terminal_bool


    def reset(self):
        return self.state_init