# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:03:29 2019

@author: Bahij
"""

class GridEnvironment:
    def __init__(self, rows, columns):
        self.rows = rows
        self.columns = columns
        self.grid = [[0] * self.columns for i in range(self.rows)]
    
    def setImpassable(self, row, column):
        self.grid[row][column]="#"
    def isImpassable(self, row, column):
        return self.grid[row][column]=="#"
    
    def setPositive(self, row, column):
        self.grid[row][column]="+"
    def isPositive(self, row, column):
        return self.grid[row][column]=="+"
    
    def setNegative(self, row, column):
        self.grid[row][column]="-"
    def isNegative(self, row, column):
        return self.grid[row][column]=="-"
    
    def setValue(self, value, row, column):
        if self.grid[row][column] > value:
            self.grid[row][column] = value
    def getValue(self, row, column):
        return self.grid[row][column]
    
    def print(self):
        for i in range(self.rows):
            print(self.grid[i])