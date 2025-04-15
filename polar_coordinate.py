import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
fig, ax = plt.subplots()


# fear = [1, 120]
# disgust = [1, 150]
# sad = [1, 210]
# neutral = [0, 0]
# happy = [1, 30]


disgust = [1, 153]
fear = [1, 117]
sad = [1, 198]
neutral = [0, 0]
happy = [1, 18]


# s = np.zeros((4, 4))
# norm_dist = np.zeros((4, 4))
# list = [neutral, sad, fear, happy]
# name = ['neutral', 'sad', 'fear', 'happy']

# for i in range(4):
#     for j in range(4):
#         # s[i,j] = 1 - math.sqrt(list[i][0]**2 + list[j][0]**2 -2*list[i][0]*list[j][0]*math.cos(math.radians(list[i][1])-math.radians(list[j][1]))) / 2

#         norm_dist[i,j] = math.sqrt(list[i][0]**2 + list[j][0]**2 -2*list[i][0]*list[j][0]*math.cos(math.radians(list[i][1])-math.radians(list[j][1])))/2
#         # Apply scaling factor beta to increase ambiguity (make distances smaller)
#         delta = 5.0
#         scaled_dist = norm_dist[i, j] ** delta


#         s[i, j] = 1 - scaled_dist    

s = np.zeros((5, 5))
norm_dist = np.zeros((5, 5))
list = [fear, disgust, sad, neutral, happy]
name = ['Fear', 'Disgust', 'Sad', 'Neutral', 'Happy']

for i in range(5):
    for j in range(5):
        # s[i,j] = 1 - math.sqrt(list[i][0]**2 + list[j][0]**2 -2*list[i][0]*list[j][0]*math.cos(math.radians(list[i][1])-math.radians(list[j][1]))) / 2

        norm_dist[i,j] = math.sqrt(list[i][0]**2 + list[j][0]**2 -2*list[i][0]*list[j][0]*math.cos(math.radians(list[i][1])-math.radians(list[j][1])))/2
        # Apply scaling factor beta to increase ambiguity (make distances smaller)
        delta = 1.0
        scaled_dist = norm_dist[i, j] ** delta


        s[i, j] = 1 - scaled_dist    


# s = np.array(s)
# # Get only the lower triangular part below the diagonal (k=-1)
# lower_bound_values = s[np.tril_indices(len(s), k=-1)]

# # Calculate the average of the lower triangular values
# average_value = np.mean(lower_bound_values)
# print('{:.2f}'.format(average_value))    
# exit(0)
# ax.matshow(s, cmap=plt.cm.Greens)
# for i in range(5):
#     for j in range(5):
#         ax.text(i, j, str(round(s[i,j], 2)), va='center', ha='center')
#
# plt.xticks([0,1,2,3,4], name)
# plt.yticks([0,1,2,3,4], name)
#
# plt.show()

confusion_matrix = sn.heatmap(s, xticklabels=name, yticklabels=name, annot=True, cmap='Blues', annot_kws={"size": 8})

plt.show()
