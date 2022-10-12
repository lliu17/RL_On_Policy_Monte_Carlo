import numpy as np
from enum import Enum
import random

class Status(Enum):
    FINISHED = 1
    FAILED = 2
    CONTINUE = 3
    IGNORE = 4

class RaceCar:
    def __init__(self, loc, track):
        self.__loc = loc
        self.__velocity = [0, 0]
        self.__track = track
        self.__values = np.zeros((track.rowNum, track.colNum))
    
    def change_velocity(self, delta, backTrack):
        prev_v0 = self.__velocity[0]
        prev_v1 = self.__velocity[1]
        self.__velocity[0] += delta[0]
        self.__velocity[1] += delta[1]
        
        if self.__velocity[0] < 0:
            self.__velocity[0] = 0

        if self.__velocity[1] < 0:
            self.__velocity[1] = 0
        
        if self.__velocity[0] > 5:
            self.__velocity[0] = 5

        if self.__velocity[1] > 5:
            self.__velocity[1] = 5

        if self.__velocity[0] == 0 and self.__velocity[1] == 0:
            print("both velocities 0, ignored")
            self.__velocity[0] = prev_v0
            self.__velocity[1] = prev_v1
            # return Status.IGNORE
        return self.update_pos(backTrack)

    def inBound(self, row, col):
        if  row >= self.__track.rowNum or\
            col >= self.__track.colNum or\
            self.__track.track[row][col] == 0:
            return False
        return True

    # note: this function assumes finish line is vertical (a column)
    def update_pos(self, backTrack):
        row = self.__loc[0]
        col = self.__loc[1]
        backTrack.append([[row, col], [self.__velocity[0], self.__velocity[1]]])
        # print("in 0: row, col =", row, col)
        # print("track value =", self.__track.track[row][col])
        # check location is part of the track if moving along velocity[0]
        for i in range(self.__velocity[0]):
            self.__loc[0] += 1
            row = self.__loc[0]
            # print("in 0: row, col =", row, col)
            if not self.inBound(row, col):
                print("moving along velocity[0] exceeds bound, should restart")
                self.calc_rewards(backTrack, OOB_PENALTY)
                return Status.FAILED
            
            # print("track value =", self.__track.track[row][col])

        # check location is part of the track if moving along velocity[1]
        # if we encounter finish line before exceeds bound, it is a success
        for i in range(self.__velocity[1]):
            self.__loc[1] += 1
            if not self.inBound(row, col):
                print("moving along velocity[1] exceeds bound, should restart")
                self.calc_rewards(backTrack, OOB_PENALTY)
                return Status.FAILED

            if self.__loc in self.__track.finish:
                print("At finish line!")
                self.calc_rewards(backTrack, GOAL_REWARD)
                return Status.FINISHED

        # neither bounds exceeded nor succeeded, continue updates
        return Status.CONTINUE

    def calc_rewards(self, backTrack, reward):
        for state, action in backTrack[::-1]:
            row = state[0]
            col = state[1]
            vel_x = action[0]
            vel_y = action[1]
            self.__values[row][col] = [vel_x, vel_y, rew]
            rew *= GAMMA

    def print(self):
        print("curr velocity =", self.__velocity)
        print("curr location =", self.__loc)
        print()

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
            for i in range(self.rowNum):
                if self.track[row_or_col][i] == 1:
                    res.append([row_or_col, i])
        elif row_or_col_str == "col":
            for i in range(self.rowNum):
                if self.track[i][row_or_col] == 1:
                    res.append([i, row_or_col])
        else:
            raise ValueError("Need to pass in row or col")

        return res

    def print(self):
        print("Racetrack:")
        print(self.track)
        print()

def getActionSet():
    return [-1, 0, 1]

def random_policy():
    actions = getActionSet()
    delta_x = random.choice(actions)
    delta_y = random.choice(actions)
    return [delta_x, delta_y]

def sim(num_episodes, racetrack, policy):
    count = 0
    random.seed(0)

    for e in range(num_episodes):
        print("\n============= Episode", e, "=============")
        start = list(random.choice(racetrack.start))
        print("start loc =", start)
        racecar = RaceCar(start, racetrack)
        backTrack = []
        res = Status.CONTINUE
        # racecar.print()

        while (res == Status.CONTINUE):
            res = racecar.change_velocity(policy(), backTrack)
            
        if res == Status.FAILED:
            print("failed")
        if res == Status.FINISHED:
            print("finished")
            count += 1
    
    print("count = ", count)


# mini = np.array([[1, 1, 1, 0, 0],
#                  [0, 1, 1, 1, 0],
#                  [0, 0, 1, 1, 1]])
GOAL_REWARD = 10
OOB_PENALTY = -10 # out of bound penalty
GAMMA = 0.9

mini = np.array([[1, 1, 1],
                 [0, 1, 1]])

miniTrack = Racetrack(mini, 0, 2)

miniTrack.print()

sim(100, miniTrack, random_policy)

