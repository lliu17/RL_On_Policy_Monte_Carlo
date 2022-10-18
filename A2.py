from audioop import reverse
from turtle import back
import numpy as np
from enum import Enum
import random
import matplotlib.pyplot as plt

GOAL_REWARD = 10
OOB_PENALTY = -100 # out of bound penalty
GAMMA = 0.9
NOISE = 0
NOT_NOISE = 0

class Status(Enum):
    FINISHED = 1
    FAILED = 2
    CONTINUE = 3

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
        print("Start:")
        print(self.start)
        print("Finish:")
        print(self.finish)
        print()

class RaceCar:
    def __init__(self, track):
        self.loc = random.choice(track.start) # init: random pos at start line
        self.__velocity = [0, 0]
        self.__track = track
        self.avgRewards = {}          # state/action pair --> average reward
        self.bestActionAtState = {}   # state --> best action, reward

    def reset(self):
        self.loc = random.choice(self.__track.start)
        self.__velocity = [0, 0]

    def inBound(self, row, col):
        if  row >= self.__track.rowNum or\
            col >= self.__track.colNum or\
            self.__track.track[row][col] == 0:
            return False
        return True

    def getBestActionAtState(self, state):
        return self.bestActionAtState[state][0]

    def getAvgReward(self, state_action):
        return self.avgRewards[state_action][0]

    def getAvgCount(self, state_action):
        return self.avgRewards[state_action][1]

    def formStateActionTuple(state, action):
        return (state[0], state[1], action[0], action[1])

    def change_velocity(self, policy, backTrack, stepNum, noise=False):
        delta = None
        if noise and random.random() < 0.2:
            global NOISE
            NOISE += 1
            delta = [0, 0]
        else:
            global NOT_NOISE
            NOT_NOISE += 1
            delta = policy(self.loc, self)

        self.__velocity[0] += delta[0]
        self.__velocity[1] += delta[1]

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

        # keep generating random delta if both x and y direction velocity 0 
        while self.__velocity[0] == 0 and self.__velocity[1] == 0:
            vel = [0, 1]
            self.__velocity[0] =  random.choice(vel)
            self.__velocity[1] =  random.choice(vel)
            delta[0] = self.__velocity[0]
            delta[1] = self.__velocity[1]
            # print("both velocities 0, regenerating delta")
        return self.update_pos(backTrack, delta, stepNum)

    # note: this function assumes finish line is vertical (a column)
    def update_pos(self, backTrack, delta, stepNum):
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
                self.calc_rewards(backTrack, stepNum, OOB_PENALTY)
                return Status.FAILED
            # print("track value =", self.__track.track[row][col])

        # check location is part of the track if moving along velocity[1]
        # if we encounter finish line before exceeds bound, it is a success
        for _ in range(vy):
            self.loc = (self.loc[0], self.loc[1] + 1)
            col = self.loc[1]
            if not self.inBound(row, col):
                # print("moving along velocity[1] exceeds bound, should restart")
                self.calc_rewards(backTrack, stepNum, OOB_PENALTY)
                return Status.FAILED

            if self.loc in self.__track.finish:
                # print("At finish line!")
                self.calc_rewards(backTrack, stepNum, GOAL_REWARD)
                return Status.FINISHED

        # neither bounds exceeded nor succeeded, continue updates
        return Status.CONTINUE

    def calc_rewards(self, backTrack, lastStep, lastStepReward):
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
                self.avgRewards[state_action] = (backTrack[state_action], 1)
                # print("0 self.avgRewards[state_action]=", self.avgRewards[state_action])
            else:
                # print("updating avg for:", state_action)
                prev_rew = self.getAvgReward(state_action)
                prev_count = self.getAvgCount(state_action)
                # print("prev_rew=", prev_rew)
                # print("prev_count=", prev_count)
                
                self.avgRewards[state_action] = \
                    ((prev_rew * prev_count + backTrack[state_action]) / (prev_count + 1),
                     prev_count + 1)
                # print("new rev=", self.avgRewards[state_action])
            
            # update best action at state and corresponding reward
            state = (state_action[0], state_action[1])
            action = (state_action[2], state_action[3])
            # if state in self.bestActionAtState:
                # print("prev best=", self.bestActionAtState[state])
            if state not in self.bestActionAtState:
                self.bestActionAtState[state] = (action, self.avgRewards[state_action])
            else:
                prev_best_action = self.getBestActionAtState(state)
                prev_best_state_action = (state_action[0], state_action[1], \
                                    prev_best_action[0], prev_best_action[1])
                # print("prev best=", prev_best_action)
                if self.getAvgReward(state_action) > self.getAvgReward(prev_best_state_action):
                    self.bestActionAtState[state] = (action, self.getAvgReward(state_action))
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

def epsilon_greedy(state, raceCar):
    rand_num = random.random()
    if rand_num < EPSILON:
        global EPSILON_COUNTER
        EPSILON_COUNTER += 1
        return random_policy(state, raceCar)
    else:
        global OPPOSITE_COUNTER
        OPPOSITE_COUNTER += 1   
        return greedy_policy(state, raceCar)

def greedy_policy(state, raceCar):
    # print("current status")
    # raceCar.print()
    delta = []
    if state in raceCar.bestActionAtState:
        # print("greedy selects", raceCar.bestActionAtState[state])
        delta = list(raceCar.getBestActionAtState(state))
        best_rew = raceCar.getAvgCount((state[0], state[1], delta[0], delta[1]))
        if np.isclose(best_rew, -100):
            print("best so far is -100, taking random action")
            delta = random_policy(state, raceCar)
    else:
        # print("state not encountered, cannot greedy select")
        delta = random_policy(state, raceCar)
    # print("delta =", delta)
    return delta

def sim(numEpisodes, raceCar, policy, noise=False):
    count = 0
    for e in range(numEpisodes):
        random.seed(e+1)
        # print("\n============= Episode", e, "=============")
        # print("start loc =", raceCar.loc)   # raceCar.print()
        res = Status.CONTINUE
        backTrack = {}
        stepNum = 0

        while (res == Status.CONTINUE):
            res = raceCar.change_velocity(policy, backTrack, stepNum, noise)
            stepNum += 1
        # print("backtrack:")
        # print(backTrack)

        if res == Status.FINISHED:
            count += 1
        raceCar.reset()
    # print("count = ", count)

def runGreedy(raceCar):
    raceCar.reset()
    # print("start loc =", raceCar.loc)
    res = Status.CONTINUE
    backTrack = {}
    stepNum = 0
    while (res == Status.CONTINUE):                         # noise is off
            res = raceCar.change_velocity(greedy_policy, backTrack, stepNum)
            stepNum += 1
            
    if (res == Status.FINISHED):
        # print("greedy success!")
        global GREEDY_SUCCESS
        GREEDY_SUCCESS += 1
    else:
        # print("greedy failed")
        global GREEDY_FAILED
        GREEDY_FAILED += 1
    
    start = list(backTrack.items())[0]   # info for first step
    print("start=", start)
    return start

EPSILON = 0.1
EPSILON_COUNTER = 0
OPPOSITE_COUNTER = 0
GREEDY_SUCCESS = 0
GREEDY_FAILED = 0
greedy_success_list_random = []
greedy_success_list_epsilon = []

def countGreedySuccess(track, lo, hi, policy, policy_name):
    global GREEDY_SUCCESS, GREEDY_FAILED
    GREEDY_SUCCESS = 0
    GREEDY_FAILED = 0
    AvgFirstStepReward = 0
    for i in range(lo, hi):
        raceCar = RaceCar(track)
        # sim(i, raceCar, policy, noise=True)
        sim(i, raceCar, policy)
        rewStart = runGreedy(raceCar)[1]
        AvgFirstStepReward = \
            (AvgFirstStepReward * (i - lo) + rewStart)/(i - lo + 1)

    print("greedy success count", GREEDY_SUCCESS)
    if policy_name == "random":
        greedy_success_list_random.append(GREEDY_SUCCESS/(GREEDY_SUCCESS + GREEDY_FAILED))
    elif policy_name == "epsilon":
        greedy_success_list_epsilon.append(GREEDY_SUCCESS/(GREEDY_SUCCESS + GREEDY_FAILED))
    else:
        raise ValueError("incorrect policy name")
     
    print("greedy failed count", GREEDY_FAILED)
    print("AvgFirstStepReward", '{:.3f}'.format(AvgFirstStepReward))
    return AvgFirstStepReward

# mini = np.array([[1, 1, 1, 0, 0],
#                  [0, 1, 1, 1, 0],
#                  [0, 0, 1, 1, 1]])
# miniTrack = Racetrack(mini, 0, 4)
# print("start set =", miniTrack.start)
# print("finish set =", miniTrack.finish)
# raceCar = RaceCar(miniTrack)
# print()
# print("counting random policy")
# countGreedySuccess(miniTrack, 10, 30, random_policy)
# print()
# print("counting epsilon greedy policy")
# countGreedySuccess(miniTrack, 10, 30, epsilon_greedy)


track1 = np.loadtxt("track1", dtype="i")
track1 = np.flipud(track1)  # last line should be the starting line (row 0)
# print("track1:\n", track1)
track1 = Racetrack(track1, 0, track1.shape[1] - 1)
raceCar = RaceCar(track1)
print("start set =", track1.start)
print("finish set =", track1.finish)
print()

startEpisode = 10
startEpisode_list = []
episodeNumRange = []
AvgFirstStepRewardList_rand = []
AvgFirstStepRewardList_epsilon = []
for i in range(6):
    print("startEpi=", startEpisode)
    startEpisode_list.append(startEpisode)
    episodeNumRange.append(startEpisode)
    print("---counting random policy")
    res = countGreedySuccess(track1, startEpisode, startEpisode + 20, random_policy, "random")
    AvgFirstStepRewardList_rand.append(res)
    print("---counting epsilon greedy policy")
    res = countGreedySuccess(track1, startEpisode, startEpisode + 20, epsilon_greedy, "epsilon")
    AvgFirstStepRewardList_epsilon.append(res)
    startEpisode *= 5
    print()

print("episodeNumRange", episodeNumRange)
print("AvgFirstStepRewardList_rand", AvgFirstStepRewardList_rand)
print("AvgFirstStepRewardList_epsilon", AvgFirstStepRewardList_epsilon)

plt.title("Average Return of Best State-Action Pair at Initial State")
plt.plot(startEpisode_list, AvgFirstStepRewardList_rand, label = "Monte Carlo ES")
plt.plot(startEpisode_list, AvgFirstStepRewardList_epsilon, label = "On-policy first-visit MC Control (Epsilon Greedy)")
plt.xlabel('Number of Episodes')
plt.ylabel('Average Return')
plt.legend()
plt.show()

plt.title("Ratio of Greedy Run Succeeds at Finishing the Track")
plt.plot(startEpisode_list, greedy_success_list_random, label = "Monte Carlo ES")
plt.plot(startEpisode_list, greedy_success_list_epsilon, label = "On-policy first-visit MC Control (Epsilon Greedy)")
plt.xlabel('Number of Episodes')
plt.ylabel('Success Ratio')
plt.legend()
plt.show()

