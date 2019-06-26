# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:03:29 2019

@author: Bahij
"""

class GridEnvironment:
    IMPASSABLE = "#"
    POSITIVE = "+"
    NEGATIVE = "-"
    
    def __init__(self, columns, rows):
        self.rows = rows
        self.columns = columns
        self.grid = [[0] * self.columns for i in range(self.rows)]
        self.positive = []
        self.negative = []
        self.impassable = []
    
    def setImpassable(self, xy):
        if xy not in self.impassable:
            self.impassable.append(xy)
        self.grid[xy[1]][xy[0]] = GridEnvironment.IMPASSABLE
    def isImpassable(self, xy):
        return self.grid[xy[1]][xy[0]] == GridEnvironment.IMPASSABLE
    
    def setPositive(self, xy):
        if xy not in self.positive:
            self.positive.append(xy)
        self.grid[xy[1]][xy[0]] = GridEnvironment.POSITIVE
    def isPositive(self, xy):
        return self.grid[xy[1]][xy[0]] == GridEnvironment.POSITIVE
    
    def setNegative(self, xy):
        if xy not in self.negative:
            self.negative.append(xy)
        self.grid[xy[1]][xy[0]] = GridEnvironment.NEGATIVE
    def isNegative(self, xy):
        return self.grid[xy[1]][xy[0]] == GridEnvironment.NEGATIVE
    
    def setValue(self, value, xy):
        if not self.isImpassable(xy) and not self.isNegative(xy) and not self.isPositive(xy) \
            and ((self.grid[xy[1]][xy[0]] == 0) or (self.grid[xy[1]][xy[0]] < value)):
            self.grid[xy[1]][xy[0]] = value
    def getValue(self, xy):
        return self.grid[xy[1]][xy[0]]

    def print(self):
        for i in range(self.rows):
            print(self.grid[i])