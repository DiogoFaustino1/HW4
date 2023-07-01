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
num_x_array = [2, 5, 11, 21]
num_y_array = [5, 11, 21, 41, 61]


## Mesh Convergence Analysis for changes in # of chordwise points

# Initialize arrays to store results for num_x variations
CDx = np.zeros(len(num_x_array))              # CD array
CDx_deltas = np.zeros(len(num_x_array)-1) 
WBMx = np.zeros(len(num_x_array))             # Wingbox Mass array
WBMx_deltas = np.zeros(len(num_x_array)-1)
Tx = np.zeros(len(num_x_array))               # CPU times array
Tx_deltas = np.zeros(len(num_x_array)-1)

# Iterates different num_x and stores values in array
for i in range(len(num_x_array)):
    start = time.time()
    CDx[i], WBMx[i] = MDA_mesh(num_x_array[i], 7)
    end = time.time()
    Tx[i] = end-start
    
# Iterates through obtained values, determines deltas (in %), and stores in array
for i in range(len(num_x_array) - 1):
    CDx_deltas[i] = ((CDx[i + 1] - CDx[i]) / CDx[i]) * 100
    WBMx_deltas[i] = ((WBMx[i + 1] - WBMx[i]) / WBMx[i]) * 100
    Tx_deltas[i] = ((Tx[i + 1] - Tx[i]) / Tx[i]) * 100
    
    
    

## Mesh Convergence Analysis for changes in # of spanwise points

# Initialize arrays to store results for num_y variations
CDy = np.zeros(len(num_y_array))              # CD array
CDy_deltas = np.zeros(len(num_y_array)-1) 
WBMy = np.zeros(len(num_y_array))             # Wingbox Mass array
WBMy_deltas = np.zeros(len(num_y_array)-1)
Ty = np.zeros(len(num_y_array))               # CPU times array
Ty_deltas = np.zeros(len(num_y_array)-1)

# Iterates different num_y and stores values in array
for i in range(len(num_y_array)):
    start = time.time()
    CDy[i], WBMy[i] = MDA_mesh(5, num_y_array[i])
    end = time.time()
    Ty[i] = end-start
    
# Iterates through obtained values, determines deltas (in %), and stores in array
for i in range(len(num_y_array) - 1):
    CDy_deltas[i] = ((CDy[i + 1] - CDy[i]) / CDy[i]) * 100
    WBMy_deltas[i] = ((WBMy[i + 1] - WBMy[i]) / WBMy[i]) * 100
    Ty_deltas[i] = ((Ty[i + 1] - Ty[i]) / Ty[i]) * 100
    
    
    

## Prints values that are copyable into LaTeX

# Guarantees all arrays (cols) are the same length
CDx_deltas = np.insert(CDx_deltas, 0, 999)    # inserts at index 0 a dummy value
WBMx_deltas = np.insert(WBMx_deltas, 0, 999)
Tx_deltas = np.insert(Tx_deltas, 0, 999) 

CDy_deltas = np.insert(CDy_deltas, 0, 999)    # inserts at index 0 a dummy value
WBMy_deltas = np.insert(WBMy_deltas, 0, 999)
Ty_deltas = np.insert(Ty_deltas, 0, 999) 


print("num_x", "CDx", "&", "CDx_deltas", "&", "WBMx", "&", "WBMx_deltas", "&",
      "Tx", "&", "Tx_deltas", "\ \\")
for i in range(len(num_x_array)):
    print(num_x_array[i], "&", CDx[i], "&", CDx_deltas[i], "&", WBMx[i], "&", WBMx_deltas[i], "&",
          Tx[i], "&", Tx_deltas[i], "\\") 
    
print("num_y", "CDy", "&", "CDy_deltas", "&", "WBMy", "&", "WBMy_deltas", "&",
      "Ty", "&", "Ty_deltas", "\ \\")
for i in range(len(num_y_array)):
    print(num_y_array[i], "&", CDy[i], "&", CDy_deltas[i], "&", WBMy[i], "&", WBMy_deltas[i], "&",
          Ty[i], "&", Ty_deltas[i], "\\")
    
    
    
# # fig, ax = plt.subplots()
# # ax.plot(num_y_values, CPU_times, marker='o')
# # ax.set_xlabel('num_y')
# # ax.set_ylabel('CPU time')
# # ax.set_title('Variation of CPU time with num_y')
# # plt.show()