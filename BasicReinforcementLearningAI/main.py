# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:28:03 2019

@author: Bahij
"""

from GridEnvironment import GridEnvironment
from GridAgent import GridAgent
from random import randint


def createRandomGrid(maxC = 15, maxR = 15):
    ge = GridEnvironment(randint(10, maxC), randint(10, maxR))
    numWalls = randint(0, int(ge.rows / ge.columns))
    numPos = randint(1, int(ge.rows/2))
    numNeg = int(numPos/2 + 1)
    for i in range(numWalls):
        xy = (randint(0, ge.columns - 1), randint(0, ge.rows - 1))
        ge.setImpassable(xy)
    for i in range(numNeg):
        xy = (randint(0, ge.columns - 1), randint(0, ge.rows - 1))
        ge.setNegative(xy)
    for i in range(numPos):
        xy = (randint(0, ge.columns - 1), randint(0, ge.rows - 1))
        ge.setPositive(xy)
    return ge

def mapGrid(ge, startPos):
    agent = GridAgent(ge, startPos)
    agent.mapBellman(iterations = ge.rows*ge.columns*500)
    return agent

def traverseGrid(agent, startPos):
    deterministic = randint(0, 100) % 2 == 0
    print("deterministic: ", deterministic)
    agent.traverse(startPos, deterministic)

def multiTraverse(agent, count = 5):
    print("grid of dimensions: %d, %d" %(agent.gridEnv.columns, agent.gridEnv.rows))
    for i in range(count):
        startPos = (randint(0, agent.gridEnv.columns - 1), randint(0, agent.gridEnv.rows - 1))
        traverseGrid(agent, startPos)

def run(numGrids = 5):
    for i in range(numGrids):
        ge = createRandomGrid()
        agent = mapGrid(ge, (0, 0))
        print()
        multiTraverse(agent)
        
run()
    