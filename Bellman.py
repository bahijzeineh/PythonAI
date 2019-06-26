# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:14:36 2019

@author: Bahij
"""
'''
V(s) = max(a){R(s,a) + gamma*V(s')}
'''

class Bellman:
    def __init__(self, discount, gridEnv):
        self.discount = discount
        self.gridEnv = gridEnv

    def getValue(self, curr):
        