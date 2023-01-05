"""
Created on Fri Oct 21 22:32:01 2022

@author: patrick
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn

"""Similarities among the five emotions placed on the emotion model"""
fig, ax = plt.subplots()

fear = [1, 117]
disgust = [1, 153]
sad = [1, 198]
neutral = [0, 0]
happy = [1, 18]


s = np.zeros((5, 5))

list = [fear, disgust, sad, neutral, happy]
name = ['Fear', 'Disgust', 'Sad', 'Neutral', 'Happy']

for i in range(5):
    for j in range(5):
        s[i,j] = 1 - math.sqrt(list[i][0]**2 + list[j][0]**2 -2*list[i][0]*list[j][0]*math.cos(math.radians(list[i][1])-math.radians(list[j][1]))) / 2


confusion_matrix = sn.heatmap(s, xticklabels=name, yticklabels=name, annot=True, cmap='Blues', annot_kws={"size": 8})
plt.show()
