from audioop import reverse
from turtle import back
import numpy as np
from enum import Enum
import random

GOAL_REWARD = 10
OOB_PENALTY = -100 # out of bound penalty
GAMMA = 0.9

class Status(Enum):
    FINISHED = 1
    FAILED = 2
    CONTINUE = 3
    IGNORE = 4

class Racetrack:
    def __init__(self, track, start, finish):
        self.track = track
        self.rowNum = track.shape[0]
        self.colNum = track.shape[1]
        self.start = self.__set_start_or_finish__("row", start)
        self.finish = self.__set_start_or_finish__("col", finish)
    
    def __set_start_or_finish__(self, row_or_col_str, row_or_col):
        res = []
        if row_or_col_str == "row":
            for i in range(self.colNum):
                if self.track[row_or_col][i] == 1:
                    res.append((row_or_col, i))
        elif row_or_col_str == "col":
            for i in range(self.rowNum):
                if self.track[i][row_or_col] == 1:
                    res.append((i, row_or_col))
        else:
            raise ValueError("Need to pass in row or col")

        return res

    def print(self):
        print("Racetrack:")
        print(self.track)
        print()

class RaceCar:
    def __init__(self, track):
        self.loc = random.choice(track.start) # init: random pos at start line
        self.__velocity = [0, 0]
        self.__track = track
        self.avgRewards = {}          # state/action pair --> average reward
        self.bestActionAtState = {}   # map: state --> best action

    def reset(self):
        self.loc = random.choice(self.__track.start)
        self.__velocity = [0, 0]

    def inBound(self, row, col):
        if  row >= self.__track.rowNum or\
            col >= self.__track.colNum or\
            self.__track.track[row][col] == 0:
            return False
        return True

    def change_velocity(self, policy, backTrack, stepNum, numEpisodes):
        # while True: # keep generating delta if both x and y direction velocity 0 
        delta = policy(self.loc, self)
        # print("0 delta=", delta)
        self.__velocity[0] += delta[0]
        self.__velocity[1] += delta[1]
        # print("0 vel=", self.__velocity[0], self.__velocity[1])

        if self.__velocity[0] < 0:
            self.__velocity[0] = 0
            delta[0] = 0

        if self.__velocity[1] < 0:
            self.__velocity[1] = 0
            delta[1] = 0
        
        if self.__velocity[0] > 5:
            self.__velocity[0] = 5
            delta[0] = 0

        if self.__velocity[1] > 5:
            self.__velocity[1] = 5
            delta[1] = 0

        while self.__velocity[0] == 0 and self.__velocity[1] == 0:
            vel = [0, 1]
            self.__velocity[0] =  random.choice(vel)
            self.__velocity[1] =  random.choice(vel)
            delta[0] = self.__velocity[0]
            delta[1] = self.__velocity[1]
            # print("both velocities 0, regenerating delta")
        # print("1 delta =", delta)
        return self.update_pos(backTrack, delta, stepNum, numEpisodes)

    # note: this function assumes finish line is vertical (a column)
    def update_pos(self, backTrack, delta, stepNum, numEpisodes):
        # print("delta in update_pos", delta)
        row = self.loc[0]
        col = self.loc[1]
        vx = self.__velocity[0]
        vy = self.__velocity[1]
        # print("loc = ", row, col, ", vel = ", vx, vy)
        dx = delta[0]
        dy = delta[1]
        if (row, col, dx, dy) not in backTrack:
            # stepNum: placeholder, will be used to determine discounted reward
            backTrack[(row, col, dx, dy)] = stepNum 
        # print("in 0: row, col =", row, col)
        # print("track value =", self.__track.track[row][col])
        # check location is part of the track if moving along velocity[0]
        for _ in range(vx):
            self.loc = (self.loc[0] + 1, self.loc[1])
            row = self.loc[0]
            # print("in 0: row, col =", row, col)
            if not self.inBound(row, col):
                # print("moving along velocity[0] exceeds bound, should restart")
                self.calc_rewards(backTrack, stepNum, numEpisodes, OOB_PENALTY)
                return Status.FAILED
            
            # print("track value =", self.__track.track[row][col])

        # check location is part of the track if moving along velocity[1]
        # if we encounter finish line before exceeds bound, it is a success
        for _ in range(vy):
            self.loc = (self.loc[0], self.loc[1] + 1)
            col = self.loc[1]
            if not self.inBound(row, col):
                # print("moving along velocity[1] exceeds bound, should restart")
                self.calc_rewards(backTrack, stepNum, numEpisodes, OOB_PENALTY)
                return Status.FAILED

            if self.loc in self.__track.finish:
                # print("At finish line!")
                self.calc_rewards(backTrack, stepNum, numEpisodes, GOAL_REWARD)
                return Status.FINISHED

        # neither bounds exceeded nor succeeded, continue updates
        return Status.CONTINUE

    def calc_rewards(self, backTrack, lastStep, episodeNum, lastStepReward):
        # sort dictionary backTrack by step, so we can correctly discount
        # backTrack = dict(sorted(backTrack.items(), key = lambda item:item[1], reverse=reverse))

        # print("\n in calc rewards:")
        # print("lastStep =", lastStep)
        for state_action, step in backTrack.items():
            # calculate discounted rewards for all steps
            # print("step =", step)
            # print("state_action=", state_action)
            if step == lastStep:
                backTrack[state_action] = lastStepReward # if last step, do no discount
                # print("0 backTrack[state_action]=", backTrack[state_action])
            else:
                backTrack[state_action] = lastStepReward * (GAMMA ** (lastStep - step))
                # print("1 backTrack[state_action]=", backTrack[state_action])

            # update average reward for state-action pair
            if state_action not in self.avgRewards:
                self.avgRewards[state_action] = backTrack[state_action]
                # print("0 self.avgRewards[state_action]=", self.avgRewards[state_action])
            else:
                self.avgRewards[state_action] = \
                    (self.avgRewards[state_action] * episodeNum \
                     + backTrack[state_action]) / (episodeNum + 1)
                # print("1 self.avgRewards[state_action]=", self.avgRewards[state_action])
            
            # update best action at state and corresponding reward
            state = (state_action[0], state_action[1])
            action = (state_action[2], state_action[3])
            # if state in self.bestActionAtState:
                # print("prev best=", self.bestActionAtState[state])
            if state in self.bestActionAtState:
                prev_best_action = self.bestActionAtState[state][0]
                prev_best = (state_action[0], state_action[1], prev_best_action[0], prev_best_action[1])
                # print("prev best=", prev_best_action)
                if self.avgRewards[state_action] > self.avgRewards[prev_best]:
                    self.bestActionAtState[state] = (action, self.avgRewards[state_action])
            else:
                self.bestActionAtState[state] = (action, self.avgRewards[state_action])
            # print("curr best=", self.bestActionAtState[state])
            # print("end: in calc rewards\n")

    def print(self):
        print("curr velocity =", self.__velocity)
        print("curr location =", self.loc)
        print()



def getActionSet(state):
    return [-1, 0, 1]

def random_policy(state, raceCar):
    actions = getActionSet(state)
    delta_x = random.choice(actions)
    delta_y = random.choice(actions)
    return [delta_x, delta_y]

def greedy_policy(state, raceCar):
    print("greedy selects", raceCar.bestActionAtState[state])
    return list(raceCar.bestActionAtState[state][0])

def sim(numEpisodes, racetrack, policy):
    count = 0
    raceCar = RaceCar(racetrack)
    print("start set =", racetrack.start)
    print("finish set =", racetrack.finish)

    for e in range(numEpisodes):
        random.seed(e)
        # print("\n============= Episode", e, "=============")
        # print("start loc =", raceCar.loc)   # raceCar.print()
        res = Status.CONTINUE
        backTrack = {}

        stepNum = 0
        while (res == Status.CONTINUE):
            res = raceCar.change_velocity(policy, backTrack, stepNum, e)
            stepNum += 1

        # print("backtrack:")
        # print(backTrack)

        # if res == Status.FAILED:
        #     print("failed, stepNum =", stepNum)
        if res == Status.FINISHED:
            # print("finished, stepNum =", stepNum)
            count += 1
        raceCar.reset()
    
    print("count = ", count)

    print("\n\n--- running greedy ---")
    print("best...")
    for item in raceCar.bestActionAtState.items():
        print(item)
    print()
    print("avg...")
    for item in raceCar.avgRewards.items():
        print(item)
    print()
    
    raceCar.reset()
    print("start loc =", raceCar.loc)
    res = Status.CONTINUE
    backTrack = {}
    stepNum = 0
    while (res == Status.CONTINUE):
            res = raceCar.change_velocity(greedy_policy, backTrack, stepNum, numEpisodes)
            stepNum += 1
    print("backtrack:")
    print(backTrack)

mini = np.array([[1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1]])
miniTrack = Racetrack(mini, 0, 4)

# mini = np.array([[1, 1, 0],
#                  [0, 1, 1]])
# miniTrack = Racetrack(mini, 0, 2)

miniTrack.print()

sim(1000, miniTrack, random_policy)

