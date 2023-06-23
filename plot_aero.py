# -*- coding: utf-8 -*-
"""
 Suplemental script to visualize optimization history of example in
 https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/aero_walkthrough.html
"""
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np

# Instantiate your CaseReader
cr = om.CaseReader("aero.db")

# Get driver cases (do not recurse to system/solver cases)
driver_cases = cr.get_cases('driver', recurse=False)

# Plot the path the design variables took to convergence
# Note that there are five lines in the left plot because "wing.twist_cp"
# contains five variables that are being optimized
var_values = []
con_values = []
obj_values = []
for case in driver_cases:
    var_values.append(case['wing.twist_cp'])
    con_values.append(case['aero_point_0.wing_perf.CL'])
    obj_values.append(case['aero_point_0.wing_perf.CD'])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Sample of possible variable/function optimization history visualization', fontsize=16)

ax1.plot(np.arange(len(var_values)), np.array(var_values))
ax1.set(xlabel='Iterations', ylabel='DV twist', title='Optimization History')
ax1.legend(['cp1','cp2','cp3','cp4','cp5'])
ax1.grid()

ax2.plot(np.arange(len(con_values)), np.array(con_values))
ax2.set(xlabel='Iterations', ylabel='Constraint', title='Optimization History')
ax2.legend(['CL'])
ax2.grid()

ax3.plot(np.arange(len(obj_values)), np.array(obj_values))
ax3.set(xlabel='Iterations', ylabel='Objective function', title='Optimization History')
ax3.legend(['CD'])
ax3.grid()

plt.show()