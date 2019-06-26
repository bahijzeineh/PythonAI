# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:22:46 2019

@author: Bahij
"""

from random import randint

class GridAgent:
    def __init__(self, gridEnv, start):
        self.gridEnv = gridEnv
        self.startPos = start
        self.currentPos = start
        self.positive = []
        self.negative = []
        self.discount = .9
    
        self.initExploredCells()
        
        self.checks = [self.canMoveUp, self.canMoveDown, self.canMoveLeft, self.canMoveRight]
        self.gets = [self.getUp, self.getDown, self.getLeft, self.getRight]
        self.actions = [self.moveUp, self.moveDown, self.moveLeft, self.moveRight]
    
    def initExploredCells(self, addRewards = False):
        self.exploredCells = [[0] * self.gridEnv.columns for i in range(self.gridEnv.rows)]
        for (x,y) in self.gridEnv.impassable:
            self.exploredCells[y][x] = 1
        if addRewards:
            for (x,y) in self.positive:
                self.exploredCells[y][x] = 1
            for (x,y) in self.negative:
                self.exploredCells[y][x] = 1
            
    
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
            self.exploredCells[pos[1]][pos[0]] += 1
            self.currentPos = pos
    def moveDown(self):
        if self.canMoveDown():
            pos = self.getDown()
            self.exploredCells[pos[1]][pos[0]] += 1
            self.currentPos = pos
    def moveLeft(self):
        if self.canMoveLeft():
            pos = self.getLeft()
            self.exploredCells[pos[1]][pos[0]] += 1
            self.currentPos = pos
    def moveRight(self):
        if self.canMoveRight():
            pos = self.getRight()
            self.exploredCells[pos[1]][pos[0]] += 1
            self.currentPos = pos
            
    def moveTo(self, pos):
        if not self.gridEnv.isImpassable(pos):
            self.exploredCells[pos[1]][pos[0]] += 1
            self.currentPos = pos
    
    def getNeighbours(self):
        nbrs=[]
        for (i, f) in enumerate(self.checks):
            if f():
                xy = self.gets[i]()
                nbrs.append(xy)
        return nbrs

    '''
    def moreToExplore(self):
        for row in self.exploredCells:
            for col in row:
                if col == 0:
                    return True
        return False
    
    def checkCurrentPosForReward(self):
        if self.gridEnv.isPositive(self.currentPos) and (self.currentPos not in self.positive):
            self.positive.append(self.currentPos)
        if self.gridEnv.isNegative(self.currentPos) and self.currentPos not in self.negative:
            self.negative.append(self.currentPos)
    
    def getLeastExplored(self, nbrs):
        lidx = -1
        lval = 100000
        for (i, n) in enumerate(nbrs):
            val = self.exploredCells[n[1]][n[0]]
            if val < lval:
                lval = val
                lidx = i
            #introduce some randomness
            elif val==lval and randint(0,100)%2:
                lval = val
                lidx = i
        return nbrs[lidx]
    
    def explore(self):
        self.moveTo(self.currentPos)
        self.checkCurrentPosForReward()
        iterations = 0
        
        while self.moreToExplore():
            neighbours = self.getNeighbours()
            if len(neighbours) == 0:
                pass
            elif len(neighbours) == 1:
                self.moveTo(neighbours[0])
                self.checkCurrentPosForReward()
            else:
                self.moveTo(self.getLeastExplored(neighbours))
                self.checkCurrentPosForReward()
            iterations += 1
        self.printExplored()
        print("%d iterations to fully explore grid." % iterations)
    '''
    def calculateBellman(self, prev, curr):
            if curr in self.gridEnv.positive:
                self.gridEnv.setValue(1.0, prev)
            elif curr in self.gridEnv.negative:
                pass
            else:
                self.gridEnv.setValue(self.discount * self.gridEnv.getValue(curr), prev)
    
    def mapBellman(self, iterations = 1500):
        self.initExploredCells()
        self.currentPos = self.startPos
        
        for i in range(iterations):
            prev = self.currentPos
            self.makeMove()
            self.calculateBellman(prev, self.currentPos)
    
    def makeMove(self):
        possible = []
        for (i, ch) in enumerate(self.checks):
            if ch():
                possible.append(i)
        decision = possible[randint(0, len(possible) - 1)]
        self.currentPos = self.gets[decision]()
    
    def traverse(self, startPos, steps = 0):
        self.moveTo(startPos)
        if self.gridEnv.getValue(startPos) == 1:
            print("arrived in %d steps." % (steps + 1))
        else:
            ll=[]
            lval = 0
            for n in self.getNeighbours():
                val = self.gridEnv.getValue(n)
                if val > lval:
                    lval = val
            for n in self.getNeighbours():
                if self.gridEnv.getValue(n) == lval:
                    ll.append(n)
            nxt = 0
            if len(ll) == 1:
                nxt = ll[0]
            else:
                nxt = ll[randint(0, len(ll) - 1)]
            self.traverse(nxt, steps + 1)

    def printExplored(self):
        for y in range(self.gridEnv.rows):
            for x in range(self.gridEnv.columns):
                out = 0
                if (x,y) == self.currentPos:
                    out = "@"
                else:
                    out = self.exploredCells[y][x]
                print(out, end = " ")
            print()
        print()
        print("positives: ", self.positive)
        print("negatives: ", self.negative)
        
                
            