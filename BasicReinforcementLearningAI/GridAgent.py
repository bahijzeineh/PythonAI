# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:22:46 2019

@author: Bahij
"""

from random import randint
from GridEnvironment import GridEnvironment


class GridAgent:
    def __init__(self, gridEnv, start):
        self.gridEnv = gridEnv
        self.startPos = start
        self.currentPos = start
        self.discount = .9
    
        self.checks = [self.canMoveUp, self.canMoveDown, self.canMoveLeft, self.canMoveRight]
        self.gets = [self.getUp, self.getDown, self.getLeft, self.getRight]
        self.actions = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
    
    def canMoveUp(self):
        return self.currentPos[1] > 0 and not self.gridEnv.isImpassable(self.getUp())
    def canMoveDown(self):
        return self.currentPos[1] < self.gridEnv.rows - 1 and not self.gridEnv.isImpassable(self.getDown())
    def canMoveLeft(self):
        return self.currentPos[0] > 0 and not self.gridEnv.isImpassable(self.getLeft())
    def canMoveRight(self):
        return self.currentPos[0] < self.gridEnv.columns - 1 and not self.gridEnv.isImpassable(self.getRight())

    def getUp(self):
        return (self.currentPos[0], self.currentPos[1] - 1)
    def getDown(self):
        return (self.currentPos[0], self.currentPos[1] + 1)
    def getLeft(self):
        return (self.currentPos[0] - 1, self.currentPos[1])
    def getRight(self):
        return (self.currentPos[0] + 1, self.currentPos[1])
    
    def moveUp(self):
        if self.canMoveUp():
            pos = self.getUp()
            self.currentPos = pos
    def moveDown(self):
        if self.canMoveDown():
            pos = self.getDown()
            self.currentPos = pos
    def moveLeft(self):
        if self.canMoveLeft():
            pos = self.getLeft()
            self.currentPos = pos
    def moveRight(self):
        if self.canMoveRight():
            pos = self.getRight()
            self.currentPos = pos
            
    def moveTo(self, pos):
        if not self.gridEnv.isImpassable(pos):
            self.currentPos = pos
    
    def getNeighbours(self):
        nbrs=[]
        for (i, f) in enumerate(self.checks):
            if f():
                xy = self.gets[i]()
                nbrs.append(xy)
        return nbrs

    def calculateBellman(self, changeValues = True):
        curr = self.currentPos
        possible = []
        for (i, ch) in enumerate(self.checks):
            if ch():
                possible.append(i)
        decision = possible[randint(0, len(possible) - 1)]
        possible = possible[0:decision] + possible[decision:]
        dpos = self.gets[decision]()
        dalts = []
        for i in possible:
            dalts.append(self.gets[i]())
        primaryProb = float(70)
        secondaryProb = float(100 - primaryProb) / len(dalts)
        chance = randint(0, 1000000) % 100
        target = 0
        
        if chance >= 100 - primaryProb:
            target = dpos
        else:
            for i in range(len(dalts)):
                if chance >= i * secondaryProb and chance < secondaryProb * (i + 1):
                    target = dalts[i]
                    break
                
        if changeValues:
            val = 0
            if target in self.gridEnv.positive:
                val = 1 * primaryProb/100
            elif target in self.gridEnv.negative:
                val = -1 * primaryProb/100
                #print("in negative: ",val)
                #pass
            else:
                val = self.gridEnv.getValue(target) * primaryProb/100
            for xy in dalts:
                if xy not in self.gridEnv.positive and xy not in self.gridEnv.negative:
                    val += self.gridEnv.getValue(xy) * secondaryProb/100
                elif xy in self.gridEnv.negative:
                    val += -1 * secondaryProb/100
                elif xy in self.gridEnv.positive:
                    val += 1 * secondaryProb/100
            self.gridEnv.setValue(self.discount * val, curr)
        return target
    
    
    def mapBellman(self, iterations = 1500):
        self.currentPos = self.startPos
        
        for i in range(iterations):
            pos = self.calculateBellman()
            if pos != 0:
                self.moveTo(pos)
            
    def traverse(self, startPos, deterministic = True, steps = 0):
        self.moveTo(startPos)
        nxt = 0
        if startPos in self.gridEnv.positive:
            print("arrived in %d steps." % steps)
        #elif startPos in self.gridEnv.negative:
         #   pass
        else:
            if deterministic:
                ll=[]
                lval = 0
                for n in self.getNeighbours():
                    val = self.gridEnv.getValue(n)
                    if val == GridEnvironment.POSITIVE:
                        lval = val
                        break
                    elif val > lval:
                        lval = val
                for n in self.getNeighbours():
                    if self.gridEnv.getValue(n) == lval:
                        ll.append(n)

                if len(ll) == 1:
                    nxt = ll[0]
                else:
                    nxt = ll[randint(0, len(ll) - 1)]
                self.traverse(nxt, deterministic, steps + 1)
            else:
                nxt = self.calculateBellman(False)
                self.traverse(nxt, deterministic, steps + 1)
