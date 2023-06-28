# -*- coding: utf-8 -*-
"""
Final Project - Mesh Convergence Study

 ========================================================================
   Instituto Superior Técnico - Aircraft Optimal Design - 2023
   
   96375 Filipe Valquaresma
   filipevalquaresma@tecnico.ulisboa.pt
   
   95782 Diogo Faustino
   diogovicentefaustino@tecnico.ulisboa.pt
 ========================================================================
"""

import numpy             as np
import matplotlib.pyplot as plt
from MDA_mesh import MDA_mesh
import time


# Define test arrays for chordwise and spanwise mesh points
num_x_array = [2, 3, 5, 7, 9, 11]
num_y_array = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

# Initialize arrays to store results
CDx = np.zeros(len(num_x_array))              # CD array
CDx_deltas = np.zeros(len(num_x_array)-1) 
WBMx = np.zeros(len(num_x_array))             # Wingbox Mass array
WBMx_deltas = np.zeros(len(num_x_array)-1)
Tx = np.zeros(len(num_x_array))               # CPU times array
Tx_deltas = np.zeros(len(num_x_array)-1)


## Mesh Convergence Analysis for changes in # of chordwise points
# Iterates different num_x and stores values in array
for i in range(len(num_x_array)):
    start = time.time()
    CDx[i], WBMx[i] = MDA_mesh(num_x_array[i],7)
    end = time.time()
    Tx[i] = end-start
    
# Iterates through obtained values, determines deltas (in %), and stores in array
for i in range(len(num_x_array) - 1):
    CDx_deltas[i] = ((CDx[i + 1] - CDx[i]) / CDx[i]) * 100
    WBMx_deltas[i] = ((WBMx[i + 1] - WBMx[i]) / WBMx[i]) * 100
    Tx_deltas[i] = ((Tx[i + 1] - Tx[i]) / Tx[i]) * 100
    
print("CD values vs. num_x:", CDx)
print("CD value differences vs. num_x:", CDx_deltas)
print("Wingbox mass values vs. num_x:", WBMx)
print("Wingbox mass value differences vs. num_x:", WBMx_deltas)
print("CPU time values vs. num_x:", Tx)
print("CPU time value differences vs. num_x:", Tx_deltas)


## Mesh Convergence Analysis for changes in # of spanwise points

# acabo amanhã lol vou dormir




# fig, ax = plt.subplots()
# ax.plot(num_y_values, CPU_times, marker='o')
# ax.set_xlabel('num_y')
# ax.set_ylabel('CPU time')
# ax.set_title('Variation of CPU time with num_y')
# plt.show()