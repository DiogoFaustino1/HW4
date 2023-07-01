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
import numpy as np
from openaerostruct.geometry.utils import generate_mesh
from openaerostruct.integration.aerostruct_groups import AerostructGeometry, AerostructPoint
from openaerostruct.structures.wingbox_fuel_vol_delta import WingboxFuelVolDelta
import openmdao.api as om
from openaerostruct.aerodynamics.lift_coeff_2D import LiftCoeff2D

# Provide coordinates for a portion of an airfoil for the wingbox cross-section as an nparray with dtype=complex (to work with the complex-step approximation for derivatives).
# These should be for an airfoil with the chord scaled to 1.
# We use the 10% to 60% portion of the NASA SC2-0612 airfoil for this case
# We use the coordinates available from airfoiltools.com. Using such a large number of coordinates is not necessary.
# The first and last x-coordinates of the upper and lower surfaces must be the same

#upper_x = np.array([0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.575,0.6], dtype="complex128")
#lower_x = np.array([0.1,0.125,0.15,0.175,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.575,0.6], dtype="complex128")
#upper_y = np.array([0.08664,0.09388,0.09993,0.10507,0.10943,0.11617,0.12074,0.12344,0.12439,0.12365,0.12112,0.11657,0.11342,0.10965], dtype="complex128")  # noqa: E201, E241
#lower_y = np.array([-0.06097,-0.06612,-0.07038,-0.07393,-0.0769,-0.0813,-0.08381,-0.08484,-0.08455,-0.08288,-0.0797,-0.07452,-0.07104,-0.06701], dtype="complex128")

upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")	
lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")	
upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538, 0.0545,  0.0551,  0.0557, 0.0563,  0.0568, 0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,    0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,    0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,   0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype="complex128")  # noqa: E201, E241	
lower_y = np.array([-0.0447, -0.046, -0.0473, -0.0485, -0.0496, -0.0506, -0.0515, -0.0524, -0.0532, -0.054, -0.0547, -0.0554, -0.056, -0.0565, -0.057, -0.0575, -0.0579, -0.0583, -0.0586, -0.0589, -0.0592, -0.0594, -0.0595, -0.0596, -0.0597, -0.0598, -0.0598, -0.0598, -0.0598, -0.0597, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0582, -0.0578, -0.0573, -0.0567, -0.0561, -0.0554, -0.0546, -0.0538, -0.0529, -0.0519, -0.0509, -0.0497, -0.0485, -0.0472, -0.0458, -0.0444], dtype="complex128")

# Create a dictionary to store options about the surface
mesh_dict = {
    "num_y": 21,
    "num_x": 5,
    "wing_type": "rect",
    "symmetry": True,
    "root_chord": 4.5
}

mesh = generate_mesh(mesh_dict)

surf_dict = {
    # Wing definition
    "name": "wing",  # give the surface some name
    "symmetry": True,  # if True, model only one half of the lifting surface
    "S_ref_type": "projected",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "mesh": mesh,
    "fem_model_type": "wingbox",  # 'wingbox' or 'tube'
    "data_x_upper": upper_x,
    "data_x_lower": lower_x,
    "data_y_upper": upper_y,
    "data_y_lower": lower_y,
    "twist_cp": np.array([0.0, 0.0]),  # [deg]
    "span": 23.24,
    "root_chord": 4.5,
    "taper": 0.3,
    "spar_thickness_cp": np.array([0.02413508, 0.04315824]),  # [m]
    "skin_thickness_cp": np.array([0.09564784, 0.15670857]),  # [m]
    "t_over_c_cp": np.array([0.12]),
    "original_wingbox_airfoil_t_over_c": 0.12,
    "sweep": 30,
    "AR": 8,
    # Aerodynamic deltas.
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    # They can be used to account for things that are not included, such as contributions from the fuselage, camber, etc.
    "CL0": 0.0,  # CL delta
    "CD0": 0.0078,  # CD delta
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": True,  # if true, compute wave drag
    # Airfoil properties for viscous drag calculation
    "k_lam": 0.03,  # fraction of chord with laminar
    # flow, used for viscous drag
    "c_max_t": 0.4,  # chordwise location of maximum thickness
    # Structural values are based on aluminum 7075
    "E": 73.1e9,  # [Pa] Young's modulus
    "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
    "yield": (420.0e6 / 1.5),  # [Pa] allowable yield stress
    "mrho": 2.78e3,  # [kg/m^3] material density
    "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
    "wing_weight_ratio": 1.25,
    "exact_failure_constraint": False,  # if false, use KS function
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    "engine_thrusts": 56400,
    "n_point_masses": 1,  # number of point masses in the system; in this case, the engine (omit option if no point masses)
    "fuel_density": 803.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
    "Wf_reserve": 1125.0  # [kg] reserve fuel mass
}

# Create a dictionary to store options about the surface
mesh_dict = {"num_y": 21, "num_x": 3, "wing_type": "rect", "symmetry": True, 
    "root_chord": 2,
    "offset": np.array([15, 0.0, 3.0])}

mesh = generate_mesh(mesh_dict)

upper_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")	
lower_x = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6], dtype="complex128")	
upper_y = np.array([ 0.0447,  0.046,  0.0472,  0.0484,  0.0495,  0.0505,  0.0514,  0.0523,  0.0531,  0.0538,  0.0545,  0.0551,  0.0557,  0.0563,  0.0568,  0.0573,  0.0577,  0.0581,  0.0585,  0.0588,  0.0591,  0.0593,  0.0595,  0.0597,  0.0599,  0.06,  0.0601,  0.0602,  0.0602,  0.0602,  0.0602,  0.0602,  0.0601,  0.06,  0.0599,  0.0598,  0.0596,  0.0594,  0.0592,  0.0589,  0.0586,  0.0583,  0.058,  0.0576,  0.0572,  0.0568,  0.0563,  0.0558,  0.0553,  0.0547,  0.0541], dtype="complex128")  # noqa: E201, E241	
lower_y = np.array([-0.0447, -0.046, -0.0472, -0.0484, -0.0495, -0.0505, -0.0514, -0.0523, -0.0531, -0.0538, -0.0545, -0.0551, -0.0557, -0.0563, -0.0568, -0.0573, -0.0577, -0.0581, -0.0585, -0.0588, -0.0591, -0.0593, -0.0595, -0.0597, -0.0599, -0.06, -0.0601, -0.0602, -0.0602, -0.0602, -0.0602, -0.0602, -0.0601, -0.06, -0.0599, -0.0598, -0.0596, -0.0594, -0.0592, -0.0589, -0.0586, -0.0583, -0.058, -0.0576, -0.0572, -0.0568, -0.0563, -0.0558, -0.0553, -0.0547, -0.0541], dtype="complex128")


surf_dict2 = {
    # Wing definition
    "name": "tail",  # give the surface some name
    "symmetry": True,  # if True, model only one half of the lifting surface
    "S_ref_type": "projected",  # how we compute the wing area,
    # can be 'wetted' or 'projected'
    "mesh": mesh,
    "span": 8.54,
    "root_chord": 2,
    "taper": 0.3,
    "sweep": 30,
    "fem_model_type": "wingbox",  # 'wingbox' or 'tube'
    "data_x_upper": upper_x,
    "data_x_lower": lower_x,
    "data_y_upper": upper_y,
    "data_y_lower": lower_y,
    "twist_cp": np.array([0.0, 0.0]),  # [deg]
    "spar_thickness_cp": np.array([0.004, 0.01]),  # [m]
    "skin_thickness_cp": np.array([0.005, 0.025]),  # [m]
    "t_over_c_cp": np.array([0.12]),
    "original_wingbox_airfoil_t_over_c": 0.12,
    #"thickness": np.array([0.08, 0.08, 0.10, 0.08]),
    # Aerodynamic deltas.
    # These CL0 and CD0 values are added to the CL and CD
    # obtained from aerodynamic analysis of the surface to get
    # the total CL and CD.
    # These CL0 and CD0 values do not vary wrt alpha.
    # They can be used to account for things that are not included, such as contributions from the fuselage, camber, etc.
    "CL0": 0.0,  # CL delta
    "CD0": 0.0078,  # CD delta
    "with_viscous": True,  # if true, compute viscous drag
    "with_wave": True,  # if true, compute wave drag
    # Airfoil properties for viscous drag calculation
    "k_lam": 0.03,  # fraction of chord with laminar
    # flow, used for viscous drag
    "c_max_t": 0.4,  # chordwise location of maximum thickness
    # Structural values are based on aluminum 7075
    "E": 73.1e9,  # [Pa] Young's modulus
    "G": (73.1e9 / 2 / 1.33),  # [Pa] shear modulus (calculated using E and the Poisson's ratio here)
    "yield": (420.0e6 / 1.5),  # [Pa] allowable yield stress
    "mrho": 2.78e3,  # [kg/m^3] material density
    "strength_factor_for_upper_skin": 1.0,  # the yield stress is multiplied by this factor for the upper skin
    "wing_weight_ratio": 1.25,    
    "struct_weight_relief": True,
    "distributed_fuel_weight": True,
    #"fuel_density": 803.0,  # [kg/m^3] fuel density (only needed if the fuel-in-wing volume constraint is used)
    "Wf_reserve": 0.0,  # [kg] reserve fuel mass
    "exact_failure_constraint": False  # if false, use KS function
}

surfaces = [surf_dict,surf_dict2]

# Create the problem and assign the model group
prob = om.Problem()

# Add problem information as an independent variables component
indep_var_comp = om.IndepVarComp()
indep_var_comp.add_output("Mach_number", val=np.array([0.78, 0.64]))
indep_var_comp.add_output("v", val=np.array([0.78 * 296.54, 0.64 * 340.294]), units="m/s")
indep_var_comp.add_output(
    "re",
    val=np.array([0.3796 * 296.54 * 0.78 * 1.0 / (1.43 * 1e-5), 1.225 * 340.294 * 0.64 * 1.0 / (1.81206 * 1e-5)]),
    units="1/m",
)
indep_var_comp.add_output("rho", val=np.array([0.3796, 1.225]), units="kg/m**3") 
indep_var_comp.add_output("speed_of_sound", val=np.array([296.54, 340.294]), units="m/s")

indep_var_comp.add_output("CT", val=0.38 / 3600, units="1/s")
indep_var_comp.add_output("R", val=3.120e6, units="m")
indep_var_comp.add_output("W0_without_point_masses", val=19731 + surf_dict["Wf_reserve"], units="kg")

indep_var_comp.add_output("load_factor", val=np.array([1.0, 2.5]))
indep_var_comp.add_output("alpha", val=0, units="deg")
indep_var_comp.add_output("alpha_maneuver", val=7.79046096, units="deg")
indep_var_comp.add_output("sweep", 30, units="deg")
indep_var_comp.add_output("span", 23.24, units="m")
indep_var_comp.add_output("tail_span", 8.54, units="m")
indep_var_comp.add_output("taper", 0.3)
indep_var_comp.add_output("tail_taper", 0.3)
prob.model.connect("sweep", "wing.sweep")
prob.model.connect("span", "wing.geometry.span")
prob.model.connect("tail_span", "tail.geometry.span")
prob.model.connect("taper", "wing.taper")
prob.model.connect("tail_taper", "tail.taper")

indep_var_comp.add_output("empty_cg", val=np.zeros((3)), units="m")

indep_var_comp.add_output("fuel_mass", val=1000.0, units="kg")

prob.model.add_subsystem("prob_vars", indep_var_comp, promotes=["*"])

point_masses = np.array([[1e3]])

point_mass_locations = np.array([[10, 2.0, 1.0]])

indep_var_comp.add_output("point_masses", val=point_masses, units="kg")
indep_var_comp.add_output("point_mass_locations", val=point_mass_locations, units="m")



# Compute the actual W0 to be used within OAS based on the sum of the point mass and other W0 weight
prob.model.add_subsystem(
    "W0", om.ExecComp("W0 = W0_without_point_masses + 2 * sum(point_masses)", units="kg"), promotes=["*"]
)

# Loop over each surface in the surfaces list
for surface in surfaces:
    # Get the surface name and create a group to contain components
    # only for this surface
    name = surface["name"]

    aerostruct_group = AerostructGeometry(surface=surface)

    # Add groups to the problem with the name of the surface.
    prob.model.add_subsystem(name, aerostruct_group)

# Loop through and add a certain number of aerostruct points
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
    prob.model.connect("load_factor", point_name + ".coupled.load_factor", src_indices=[i])

    
    for surface in surfaces:
        name = surface["name"]
        
        com_name = point_name + "." + name + "_perf."
        prob.model.connect(
            name + ".local_stiff_transformed", point_name + ".coupled." + name + ".local_stiff_transformed"
        )
        prob.model.connect(name + ".nodes", point_name + ".coupled." + name + ".nodes")

        # Connect aerodynamic mesh to coupled group mesh
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
        if name == "wing":
            prob.model.connect("point_masses", coupled_name + ".point_masses")
            prob.model.connect("point_mass_locations", coupled_name + ".point_mass_locations")

prob.model.add_subsystem("Cl", LiftCoeff2D(surface=surf_dict), promotes_outputs=["Cl"])
prob.model.connect("AS_point_0.coupled.aero_states.wing_sec_forces", "Cl.sec_forces")
prob.model.connect("AS_point_0.coupled.wing.widths", "Cl.widths")
prob.model.connect("AS_point_0.coupled.wing.lengths", "Cl.chords")
prob.model.promotes("Cl", inputs=["alpha"])
prob.model.promotes("Cl", inputs=["rho"], src_indices=([0]))
prob.model.promotes("Cl", inputs=["v"], src_indices=([0]))

prob.model.connect("alpha", "AS_point_0" + ".alpha")
prob.model.connect("alpha_maneuver", "AS_point_1" + ".alpha")

prob.model.add_subsystem("fuel_vol_delta", WingboxFuelVolDelta(surface=surf_dict))
prob.model.connect("wing.struct_setup.fuel_vols", "fuel_vol_delta.fuel_vols")
prob.model.connect("AS_point_0.fuelburn", "fuel_vol_delta.fuelburn")

if surf_dict["distributed_fuel_weight"]:
    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_0.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_0.coupled.wing.struct_states.fuel_mass")

    prob.model.connect("wing.struct_setup.fuel_vols", "AS_point_1.coupled.wing.struct_states.fuel_vols")
    prob.model.connect("fuel_mass", "AS_point_1.coupled.wing.struct_states.fuel_mass")

comp = om.ExecComp("fuel_diff = (fuel_mass - fuelburn) / fuelburn", units="kg")
prob.model.add_subsystem("fuel_diff", comp, promotes_inputs=["fuel_mass"], promotes_outputs=["fuel_diff"])
prob.model.connect("AS_point_0.fuelburn", "fuel_diff.fuelburn")


#############################################################################################################################################

prob.model.add_objective("AS_point_0.fuelburn", scaler=1e-5)
#prob.model.add_objective("AS_point_0.CD")
#prob.model.add_design_var("wing.twist_cp", lower=np.array([[0, 0]]), upper=np.array([10, 10]), scaler=0.1)
prob.model.add_design_var("tail.twist_cp", lower=np.array([[-10, -10]]), upper=np.array([[10, 10]]), scaler=0.1)
#prob.model.add_design_var("wing.spar_thickness_cp", lower=0.003, upper=0.5, scaler=1e2)
#prob.model.add_design_var("wing.skin_thickness_cp", lower=0.003, upper=0.5, scaler=1e2)
#prob.model.add_design_var("wing.sweep", lower=0, upper=40)
#prob.model.add_design_var("wing.geometry.span", lower=15, upper=30, scaler=0.1)
#prob.model.add_design_var("tail.geometry.span", lower=6, upper=12, scaler=0.1)
#prob.model.add_design_var("wing.taper", lower=0.1, upper=0.5)
#prob.model.add_design_var("tail.taper", lower=0.1, upper=0.5)
prob.model.add_design_var("alpha", lower=0.0, upper=15)
prob.model.add_design_var("alpha_maneuver", lower=0.0, upper=15)
#prob.model.add_design_var("point_mass_locations", lower=np.array([[0, 2.0, 1.0]]), upper=np.array([[8, 10.0, 4.0]]))

prob.model.add_constraint("AS_point_0.CM", lower=0.0, upper= 0.001)
#prob.model.add_constraint("AS_point_0.L_equals_W", equals= 0.0)
prob.model.add_constraint("AS_point_1.L_equals_W", equals= 0.0)

#prob.model.add_constraint("AS_point_1.wing_perf.failure", upper=0.0)

#prob.model.add_constraint("fuel_vol_delta.fuel_vol_delta", lower=0.0)

#prob.model.add_constraint("Cl", upper=0.6)
#prob.model.add_design_var("fuel_mass", lower=0.0, upper=2e5, scaler=1e-5)
#prob.model.add_constraint("fuel_diff", equals=0.0)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP" #['SLSQP', 'trust-constr', 'Nelder-Mead']
prob.driver.options["tol"] = 1e-9
#prob.driver.options["maxiter"] = 10000

recorder = om.SqliteRecorder("aerostruct.db")
prob.driver.add_recorder(recorder)

# We could also just use prob.driver.recording_options['includes']=['*'] here, but for large meshes the database file becomes extremely large. So we just select the variables we need.
prob.driver.recording_options["includes"] = ['*']

prob.driver.recording_options["record_objectives"] = True
prob.driver.recording_options["record_constraints"] = True
prob.driver.recording_options["record_desvars"] = True
prob.driver.recording_options["record_inputs"] = True

# Set up the problem
prob.setup()

# change linear solver for aerostructural coupled adjoint
prob.model.AS_point_0.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=30, use_aitken=True)
prob.model.AS_point_1.coupled.linear_solver = om.LinearBlockGS(iprint=0, maxiter=30, use_aitken=True)

#om.view_model(prob)

#prob.check_partials(form='central', compact_print=True, show_only_incorrect=True)

prob.run_driver()

print("The fuel burn value is", prob["AS_point_0.fuelburn"][0], "[kg]")
print(
    "The wingbox mass (excluding the wing_weight_ratio) is",
    prob["wing.structural_mass"][0] / surf_dict["wing_weight_ratio"],
    "[kg]",
)

# Output the results
print("alpha =", prob["alpha"])
print("alpha 2.5g =", prob["alpha_maneuver"])
print("sweep =", prob["wing.geometry.sweep"])
print("span =", prob["wing.geometry.span"])
print("tail span =", prob["tail.geometry.span"])
print("thickness over chord =", prob["wing.geometry.t_over_c_cp"])
print("twist_cp =", prob["wing.twist_cp"])
print("tail twist_cp =", prob["tail.twist_cp"])
print("spar thickness =", prob["wing.spar_thickness_cp"])
print("skin thickness =", prob["wing.skin_thickness_cp"])
print("point mass locations =", prob["point_mass_locations"])
print("C_D =", prob["AS_point_0.wing_perf.CD"])
print("C_L =", prob["AS_point_0.wing_perf.CL"])
print("tail C_D =", prob["AS_point_0.tail_perf.CD"])
print("tail C_L =", prob["AS_point_0.tail_perf.CL"])
print("CM vector =", prob["AS_point_0.CM"])
print("CG vector =", prob["AS_point_0.cg"])
print("Cl of sections =", prob["Cl"])
#print("Forces on sections =", prob["AS_point_0.coupled.aero_states.wing_sec_forces"])
#print("Forces on tail sections =", prob["AS_point_0.coupled.aero_states.tail_sec_forces"])
print("AS_point_0.L_equals_W =", prob["AS_point_0.L_equals_W"])
print("AS_point_1.L_equals_W =", prob["AS_point_1.L_equals_W"])
print("chord =", prob["AS_point_0.coupled.wing.lengths"])
print("tail chord =", prob["AS_point_0.coupled.tail.lengths"])
# Clean up
prob.cleanup()