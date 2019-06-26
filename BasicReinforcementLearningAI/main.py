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

agent = GridAgent(ge,(0, 0))

ge.print()
#agent.explore()
agent.mapBellman()

ge.print()

agent.traverse((2, 2))
agent.traverse((4, 5))
agent.traverse((4, 6), False)
agent.traverse((4, 4), False)
