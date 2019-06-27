# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 16:28:03 2019

@author: Bahij
"""

from GridEnvironment import GridEnvironment
from GridAgent import GridAgent


ge = GridEnvironment(5, 9)
ge.setPositive((4, 8))
ge.setPositive((0,2))
ge.setImpassable((0, 1))
ge.setImpassable((3, 2))
ge.setNegative((0, 6))
ge.setNegative((1, 8))

agent = GridAgent(ge,(0, 0))

ge.print()

agent.mapBellman()

ge.print()
print()

print("deterministic:")
agent.traverse((2, 2))
agent.traverse((4, 5))
print("non deterministic:")
agent.traverse((4, 6), deterministic = False)
agent.traverse((4, 4), deterministic = False)
