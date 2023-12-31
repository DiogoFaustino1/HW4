# -*- coding: utf-8 -*-
"""
 Suplemental script to visualize optimization history of example in
 https://mdolab-openaerostruct.readthedocs-hosted.com/en/latest/aerostructural_tube_walkthrough.html
"""
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np

# Instantiate your CaseReader
cr = om.CaseReader("aerostruct.db")

# Get driver cases (do not recurse to system/solver cases)
driver_cases = cr.get_cases('driver', recurse=False)

# Plot the path the design variables took to convergence
# Note that there are five lines in the left plot because "wing.twist_cp"
# contains five variables that are being optimized
va1_values = []
va2_values = []
co1_values = []
co2_values = []
va3_values = []
co3_values = []
obj_values = []
for case in driver_cases:
    va1_values.append(case['alpha']) # 5
    va2_values.append(case['alpha_maneuver']) # 3
    co1_values.append(case['wing.geometry.span'])
    co2_values.append(case['wing.twist_cp']) # 3
    va3_values.append(case['sweep_constraint.sweep_times_span'])
    co3_values.append(case['AS_point_0.fuelburn'])
    obj_values.append(case['wing.sweep'])

fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(1, 7)
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

fig.suptitle('Sample of possible variable/function optimization history visualization', fontsize=16)

ax1.plot(np.arange(len(va1_values)), np.array(va1_values))
ax1.set(xlabel='Iterations', ylabel='alpha', title='Optimization History')
#ax1.legend(['cp1','cp2','cp3','cp4','cp5'])
ax1.grid()

ax2.plot(np.arange(len(va2_values)), np.array(va2_values))
ax2.set(xlabel='Iterations', ylabel='alpha maneuver', title='Optimization History')
#ax2.legend(['cp1','cp2','cp3'])
ax2.grid()

ax3.plot(np.arange(len(co1_values)), np.array(co1_values))
ax3.set(xlabel='Iterations', ylabel='Wing span', title='Optimization History')
#ax3.legend(['cp1','cp2'])
ax3.grid()

ax4.plot(np.arange(len(co2_values)), np.array(co2_values))
ax4.set(xlabel='Iterations', ylabel='Wing twist', title='Optimization History')
ax4.legend(['Twist @ Tip', 'Twist @ Root'])
ax4.grid()

ax5.plot(np.arange(len(va3_values)), np.array(va3_values))
ax5.set(xlabel='Iterations', ylabel='sweep_times_span Constraint', title='Optimization History')
#ax5.legend(['Thickness Tip', 'Thickness Root'])
ax5.grid()

ax6.plot(np.arange(len(co3_values)), np.array(co3_values))
ax6.set(xlabel='Iterations', ylabel='Objective function', title='Optimization History')
ax6.legend(['fuelburn'])
ax6.grid()

ax7.plot(np.arange(len(obj_values)), np.array(obj_values))
ax7.set(xlabel='Iterations', ylabel='Wing sweep', title='Optimization History')
ax7.legend(['Sweep'])
ax7.grid()

plt.show()