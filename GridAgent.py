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
        self.positive=[]
        self.negative=[]
        
        self.exploredCells = [[0] * gridEnv.columns for i in range(gridEnv.rows)]
        for (x,y) in self.gridEnv.impassable:
            self.exploredCells[y][x] = 1
            
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
    
    def getUnexploredNeighbours(self):
        nbrs=[]
        for (i, f) in enumerate(self.checks):
            if f():
                xy = self.gets[i]()
                nbrs.append(self.gets[i]())
                self.exploredCells[xy[1]][xy[0]] += 1
        return nbrs

    def moreToExplore(self):
        for row in self.exploredCells:
            for col in row:
                if col == 0:
                    return True
        return False
    
    def explore(self):
        iterations = 0
        while self.moreToExplore():
            #self.printExplored()
            neighbours = self.getUnexploredNeighbours()
            if len(neighbours) == 0:
                pass
            elif len(neighbours) == 1:
                self.moveTo(neighbours[0])
            else:
                self.moveTo(neighbours[randint(0, len(neighbours) - 1)])
            if self.gridEnv.isPositive(self.currentPos) and (self.currentPos not in self.positive):
                self.positive.append(self.currentPos)
            if self.gridEnv.isNegative(self.currentPos) and self.currentPos not in self.negative:
                self.negative.append(self.currentPos)
            iterations += 1
        self.printExplored()
        print("%d iterations to fully explore grid." % iterations)
        
    def printExplored(self):
        for y in range(self.gridEnv.rows):
            for x in range(self.gridEnv.columns):
                out = 0
                if (x,y) == self.currentPos:
                    out = "@"
                else:
                    out = self.exploredCells[y][x]
                print(out,end = " ")
            print()
        print()
        print("positives: ", self.positive)
        print("negatives: ", self.negative)
        
                
            