# -*- coding: utf-8 -*-
"""
 Suplemental script to visualize optimization history of example in
 https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/struct_example.html
"""
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np

# Instantiate your CaseReader
cr = om.CaseReader("struct.db")

# Get driver cases (do not recurse to system/solver cases)
driver_cases = cr.get_cases('driver', recurse=False)

# Plot the path the design variables took to convergence
# Note that there are three lines in the left plot because "wing.thickness_cp"
# contains three variables that are being optimized
var_values = []
co1_values = []
co2_values = []
obj_values = []
for case in driver_cases:
    var_values.append(case['wing.thickness_cp']) # 3
    co1_values.append(case['wing.failure'])
    co2_values.append(case['wing.thickness_intersects']) # 3
    obj_values.append(case['wing.structural_mass'])

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Sample of possible variable/function optimization history visualization', fontsize=16)

ax1.plot(np.arange(len(var_values)), np.array(var_values))
ax1.set(xlabel='Iterations', ylabel='DV thickness', title='Optimization History')
ax1.legend(['cp1','cp2','cp3'])
ax1.grid()

ax2.plot(np.arange(len(co1_values)), np.array(co1_values))
ax2.set(xlabel='Iterations', ylabel='Constraint', title='Optimization History')
ax2.legend(['failure'])
ax2.grid()

ax3.plot(np.arange(len(co2_values)), np.array(co2_values))
ax3.set(xlabel='Iterations', ylabel='Constraint', title='Optimization History')
ax3.legend(['intersect1','intersect2','intersect3'])
ax3.grid()

ax4.plot(np.arange(len(obj_values)), np.array(obj_values))
ax4.set(xlabel='Iterations', ylabel='Objective function', title='Optimization History')
ax4.legend(['mass'])
ax4.grid()

plt.show()