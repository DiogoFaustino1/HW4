# -*- coding: utf-8 -*-
"""
Assignment 3 - Problem 3 d) i) 

 ========================================================================
   Instituto Superior Técnico - Aircraft Optimal Design - 2023
   
   96375 Filipe Valquaresma
   filipevalquaresma@tecnico.ulisboa.pt
   
   95782 Diogo Faustino
   diogovicentefaustino@tecnico.ulisboa.pt
 ========================================================================
"""

import numpy        as np
import openmdao.api as om
from openaerostruct.geometry.utils           import generate_mesh
from openaerostruct.geometry.geometry_group  import Geometry
from openaerostruct.aerodynamics.aero_groups import AeroPoint
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta
from sweep_over_span                    import SweepTimesSpan

# Create a dictionary to store options about the mesh
mesh_dict = {
    "num_y": 15,
    "num_x": 3,
    "wing_type": "uCRM_based",
    "symmetry": True,
    "chord_cos_spacing": 0,
    "span_cos_spacing": 0,
    "num_twist_cp": 4,
}

# Generate the aerodynamic mesh based on the previous dictionary

mesh, twist_cp = generate_mesh(mesh_dict)

upper_x = np.array([0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6], dtype="complex128")
lower_x = np.array([0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6], dtype="complex128")
upper_y = np.array([0.086640, 0.093880, 0.099930, 0.105070, 0.109430, 0.116170, 0.120740, 0.123440, 0.124390, 0.123650, 0.121120, 0.116570, 0.113420, 0.109650], dtype="complex128")  # noqa: E201, E241
lower_y = np.array([-0.060970, -0.066120, -0.070380, -0.073930, -0.076900, -0.081300, -0.083810, -0.084840, -0.084550, -0.082880, -0.079700, -0.074520, -0.071040, -0.067010], dtype="complex128")
# fmt: on

# Create a dictionary with info and options about the aerodynamic lifting surface
surface = {
    # Wing definition
    "name": "wing",  # name of the surface
    "symmetry": True,  # if true, model one half of wing reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area, can be 'wetted' or 'projected' ACHO QUE ISTO É MAIS ACCURATE DO QUE O PROJECTED
    "span": 23.24,
    "root_chord": (1.0),
    "fem_model_type": "wingbox", #ACHO QUE ISTO É MAIS ACCURATE DO QUE O TUBE
    "data_x_upper": upper_x,
    "data_x_lower": lower_x,
    "data_y_upper": upper_y,
    "data_y_lower": lower_y,
    "spar_thickness_cp": np.array([0.004, 0.004, 0.004, 0.004]),  # [m]
    "skin_thickness_cp": np.array([0.003, 0.006, 0.010, 0.012]),  # [m]
    "t_over_c_cp": np.array([0.08, 0.08, 0.10, 0.08]),
    "original_wingbox_airfoil_t_over_c": 0.12,
    "sweep": 30,
    "taper": 0.3,
    "AR": 8,
    "twist_cp": np.array([4.0, 5.0, 8.0, 9.0]),  # [deg]
    "mesh": mesh,
    #"chord_cp" : np.ones(10),
    # Aerodynamic performance of the lifting surface at an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD obtained from aerodynamic analysis of the surface
    # to get the total CL and CD. These CL0 and CD0 values do not vary with alpha.
    "CL0": 0.0,  # CL of the surface at alpha=0
    "CD0": 0.015,  # CD of the surface at alpha=0
 
    # Airfoil properties for viscous drag calculation VAMOS USAR A MERDA DO NASA LS(1)-0421
    "k_lam": 0.05,  # percentage of chord with laminar flow, used for viscous drag
    #"t_over_c_cp": np.array([0.12, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]), # thickness over chord ratio
    "c_max_t": 0.4,  # chordwise location of maximum thickness
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": False,  # if true, compute wave drag

    # Structural values are based on aluminum 7075
    "E": 73.1e9,  # [Pa] Young's modulus
    "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
    "yield": (420.0e6 / 2.5),  # [Pa] allowable yield stress
    "mrho": 2.78e3,  # [kg/m^3] material density
    "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
    "wing_weight_ratio": 1.25,
    "exact_failure_constraint": False,  # if false, use KS function
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    "n_point_masses": 1,  # number of point masses in the system; in this case, the engine (omit option if no point masses)
    "fuel_density": 803.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
    "Wf_reserve": 500.0,  # [kg] reserve fuel mass
}

mesh_dict = {"num_y": 7, "num_x": 2, "wing_type": "rect", "symmetry": True, "offset": np.array([50, 0.0, 0.0])}

mesh = generate_mesh(mesh_dict)

surface_tail = {
    # Wing definition
    "name": "tail",  # name of the surface
    "symmetry": True,  # if true, model one half of wing reflected across the plane y = 0
    "S_ref_type": "wetted",  # how we compute the wing area, can be 'wetted' or 'projected' ACHO QUE ISTO É MAIS ACCURATE DO QUE O PROJECTED
    "span": 2,
    "root_chord": 0.4,
    "fem_model_type": "tube", #ACHO QUE ISTO É MAIS ACCURATE DO QUE O TUBE
    "thickness_cp": np.array([0.1, 0.1, 0.1, 0.1]),  # [m]
    "t_over_c_cp": np.array([0.1, 0.1, 0.10, 0.1]),
    #"original_wingbox_airfoil_t_over_c": 0.12,
    #"sweep": 30,
    #"taper": 0.3,
    #"AR": 8,
    #"twist_cp": np.zeros(10),
    "mesh": mesh,
    #"chord_cp" : np.ones(10),
    # Aerodynamic performance of the lifting surface at an angle of attack of 0 (alpha=0).
    # These CL0 and CD0 values are added to the CL and CD obtained from aerodynamic analysis of the surface
    # to get the total CL and CD. These CL0 and CD0 values do not vary with alpha.
    "CL0": 0.0,  # CL of the surface at alpha=0
    "CD0": 0.0,  # CD of the surface at alpha=0
 
    # Airfoil properties for viscous drag calculation VAMOS USAR A MERDA DO NASA LS(1)-0421
    "k_lam": 0.05,  # percentage of chord with laminar flow, used for viscous drag
    #"t_over_c_cp": np.array([0.12, 0.08, 0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03]), # thickness over chord ratio
    "c_max_t": 0.4,  # chordwise location of maximum thickness
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": False,  # if true, compute wave drag

    # Structural values are based on aluminum 7075
    "E": 73.1e9,  # [Pa] Young's modulus
    "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
    "yield": (420.0e6 / 2.5),  # [Pa] allowable yield stress
    "mrho": 2.78e3,  # [kg/m^3] material density
    "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
    "fem_origin": 0.35,
    #"wing_weight_ratio": 1.25,
    "exact_failure_constraint": False,  # if false, use KS function
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    #"n_point_masses": 1,  # number of point masses in the system; in this case, the engine (omit option if no point masses)
    #"fuel_density": 803.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
    #"Wf_reserve": 500.0,  # [kg] reserve fuel mass
}

surfaces = [surface,surface_tail]

# Create the OpenMDAO problem
prob = om.Problem()

# Create an independent variable component that will supply the flow conditions to the problem
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("v", val=240, units="m/s")
indep_var_comp.add_output("Mach_number", val=0.78)
indep_var_comp.add_output("alpha", val=5.0, units="deg")
indep_var_comp.add_output("rho", val=1.00649, units="kg/m**3")
indep_var_comp.add_output("cg", val=np.zeros((3)), units="m")
indep_var_comp.add_output("sweep", 30, units="deg")
indep_var_comp.add_output("span", 23.24, units="m")
indep_var_comp.add_output("AR", 8)
#indep_var_comp.add_output("sweep_times_span", 0, units="deg*m")

# Add the independent variable component to the problem model
prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

# Create and add a group that handles the geometry for the aerodynamic lifting surface
#geom_group = Geometry(surface=surface)
#prob.model.add_subsystem(surface["name"], geom_group)

# Loop over each surface in the surfaces list
for i in surfaces:
    # Get the surface name and create a group to contain components
    # only for this surface
    name = i["name"]

    aerostruct_group = AerostructGeometry(surface=i)

    # Add groups to the problem with the name of the surface.
    prob.model.add_subsystem(name, aerostruct_group)

# Create the aero point group, which contains the actual aerodynamic analyses
for i in range(2):
    point_name = "AS_point_{}".format(i)
    # Connect the parameters within the model for each aero point

    # Create the aerostruct point group and add it to the model
    AS_point = AerostructPoint(surfaces=surfaces, internally_connect_fuelburn=False)

    prob.model.add_subsystem(point_name, AS_point)

    # Connect flow properties to the analysis point
    prob.model.connect("v", point_name + ".v", src_indices=[i])
    prob.model.connect("Mach_number", point_name + ".Mach_number", src_indices=[i])
    prob.model.connect("re", point_name + ".re", src_indices=[i])
    prob.model.connect("rho", point_name + ".rho", src_indices=[i])
    prob.model.connect("CT", point_name + ".CT")
    prob.model.connect("R", point_name + ".R")
    prob.model.connect("W0", point_name + ".W0")
    prob.model.connect("speed_of_sound", point_name + ".speed_of_sound", src_indices=[i])
    prob.model.connect("empty_cg", point_name + ".empty_cg")
    prob.model.connect("load_factor", point_name + ".load_factor", src_indices=[i])
    prob.model.connect("fuel_mass", point_name + ".total_perf.L_equals_W.fuelburn")
    prob.model.connect("fuel_mass", point_name + ".total_perf.CG.fuelburn")

    for i in surfaces:
        name = i["name"]

        if surface["distributed_fuel_weight"]:
            prob.model.connect("load_factor", point_name + ".coupled.load_factor", src_indices=[i])

        com_name = point_name + "." + name + "_perf."
        prob.model.connect(
            name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed"
        )
        prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

        # Connect aerodyamic mesh to coupled group mesh
        prob.model.connect(name + ".mesh", point_name + ".coupled." + name + ".mesh")
        if surface["struct_weight_relief"]:
            prob.model.connect(name + ".element_mass", point_name + ".coupled." + name + ".element_mass")

        # Connect performance calculation variables
        prob.model.connect(name + ".nodes", com_name + "nodes")
        prob.model.connect(name + ".cg_location", point_name + "." + "total_perf." + name + "_cg_location")
        prob.model.connect(name + ".structural_mass", point_name + "." + "total_perf." + name + "_structural_mass")

        # Connect wingbox properties to von Mises stress calcs
        prob.model.connect(name + ".Qz", com_name + "Qz")
        prob.model.connect(name + ".J", com_name + "J")
        prob.model.connect(name + ".A_enc", com_name + "A_enc")
        prob.model.connect(name + ".htop", com_name + "htop")
        prob.model.connect(name + ".hbottom", com_name + "hbottom")
        prob.model.connect(name + ".hfront", com_name + "hfront")
        prob.model.connect(name + ".hrear", com_name + "hrear")

        prob.model.connect(name + ".spar_thickness", com_name + "spar_thickness")
        prob.model.connect(name + ".t_over_c", com_name + "t_over_c")

        coupled_name = point_name + ".coupled." + name
        prob.model.connect("point_masses", coupled_name + ".point_masses")
        prob.model.connect("point_mass_locations", coupled_name + ".point_mass_locations")
# Add the SweepTimesSpan constraint component to the problem model
prob.model.add_subsystem("sweep_constraint", SweepTimesSpan(), promotes_inputs=["sweep", "span"], promotes_outputs=["sweep_times_span"])

# Connect the necessary variables to the SweepTimesSpan component

# prob.model.connect("wing.mesh", "sweep_constraint.mesh")
prob.model.connect("sweep", "wing.sweep")
prob.model.connect("span", "wing.span")

# Add the SweepTimesSpan constraint to the problem
prob.model.add_constraint("sweep_times_span", lower=0, upper=1)

# Add the design variables, constraint, and objective to the problem
prob.model.add_design_var("wing.chord_cp", lower=0.5, upper=1)
prob.model.add_design_var("wing.twist_cp", lower=-15.0, upper=15.0)
prob.model.add_design_var("alpha", lower=-50.0, upper=50.0)
prob.model.add_design_var("wing.sweep", lower=0, upper=40)
prob.model.add_design_var("wing.span", lower=0.1, upper=30)
prob.model.add_constraint("aero_point_0.wing_perf.CL", equals=0.5)
prob.model.add_objective("aero_point_0.wing_perf.CD", scaler=1e4)

# Set up the optimization problem
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["tol"] = 1e-9

recorder = om.SqliteRecorder("aero.db")
prob.driver.add_recorder(recorder)
prob.driver.recording_options['record_derivatives'] = True
prob.driver.recording_options["record_constraints"] = True
prob.driver.recording_options["record_desvars"] = True
prob.driver.recording_options["record_inputs"] = True
prob.driver.recording_options['includes'] = ['*']

prob.setup()

# Run the optimization
prob.run_driver()

# Output the results
print("alpha =", prob["alpha"])
print("sweep =", prob["wing.sweep"])
print("span =", prob["wing.span"])
print("C_D =", prob["aero_point_0.wing_perf.CD"])
print("C_L =", prob["aero_point_0.wing_perf.CL"])
print("CM position =", prob["aero_point_0.CM"][1])

# Clean up
prob.cleanup()
