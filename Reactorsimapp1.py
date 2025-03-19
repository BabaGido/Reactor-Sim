import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Title of the app
st.title("Hydrocracking and Isomerization Reactor Simulator")

# Reaction library for C15 to C20
reaction_library = {
    # Hydrocracking reactions
    "Hydrocracking_C20": {
        "reactants": ["C20H42"],
        "products": ["C10H22", "C10H20"],
        "rate_constant": 1.0,  # Base rate constant
        "activation_energy": 50.0,  # kJ/mol
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.2, "surface_area": 200},  # m²/g
            "Catalyst B": {"activity": 0.8, "surface_area": 150},
        },
    },
    "Hydrocracking_C19": {
        "reactants": ["C19H40"],
        "products": ["C9H20", "C10H20"],
        "rate_constant": 0.95,
        "activation_energy": 48.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.2, "surface_area": 200},
            "Catalyst B": {"activity": 0.8, "surface_area": 150},
        },
    },
    "Hydrocracking_C18": {
        "reactants": ["C18H38"],
        "products": ["C9H20", "C9H18"],
        "rate_constant": 0.9,
        "activation_energy": 46.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.2, "surface_area": 200},
            "Catalyst B": {"activity": 0.8, "surface_area": 150},
        },
    },
    "Hydrocracking_C17": {
        "reactants": ["C17H36"],
        "products": ["C8H18", "C9H18"],
        "rate_constant": 0.85,
        "activation_energy": 44.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.2, "surface_area": 200},
            "Catalyst B": {"activity": 0.8, "surface_area": 150},
        },
    },
    "Hydrocracking_C16": {
        "reactants": ["C16H34"],
        "products": ["C8H18", "C8H16"],
        "rate_constant": 0.8,
        "activation_energy": 42.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.2, "surface_area": 200},
            "Catalyst B": {"activity": 0.8, "surface_area": 150},
        },
    },
    "Hydrocracking_C15": {
        "reactants": ["C15H32"],
        "products": ["C7H16", "C8H16"],
        "rate_constant": 0.75,
        "activation_energy": 40.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.2, "surface_area": 200},
            "Catalyst B": {"activity": 0.8, "surface_area": 150},
        },
    },
    # Isomerization reactions
    "Isomerization_C20": {
        "reactants": ["C20H42"],
        "products": ["iC20H42"],
        "rate_constant": 0.5,
        "activation_energy": 40.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.1, "surface_area": 200},
            "Catalyst B": {"activity": 0.9, "surface_area": 150},
        },
    },
    "Isomerization_C19": {
        "reactants": ["C19H40"],
        "products": ["iC19H40"],
        "rate_constant": 0.48,
        "activation_energy": 38.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.1, "surface_area": 200},
            "Catalyst B": {"activity": 0.9, "surface_area": 150},
        },
    },
    "Isomerization_C18": {
        "reactants": ["C18H38"],
        "products": ["iC18H38"],
        "rate_constant": 0.46,
        "activation_energy": 36.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.1, "surface_area": 200},
            "Catalyst B": {"activity": 0.9, "surface_area": 150},
        },
    },
    "Isomerization_C17": {
        "reactants": ["C17H36"],
        "products": ["iC17H36"],
        "rate_constant": 0.44,
        "activation_energy": 34.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.1, "surface_area": 200},
            "Catalyst B": {"activity": 0.9, "surface_area": 150},
        },
    },
    "Isomerization_C16": {
        "reactants": ["C16H34"],
        "products": ["iC16H34"],
        "rate_constant": 0.42,
        "activation_energy": 32.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.1, "surface_area": 200},
            "Catalyst B": {"activity": 0.9, "surface_area": 150},
        },
    },
    "Isomerization_C15": {
        "reactants": ["C15H32"],
        "products": ["iC15H32"],
        "rate_constant": 0.4,
        "activation_energy": 30.0,
        "catalyst_effects": {
            "Catalyst A": {"activity": 1.1, "surface_area": 200},
            "Catalyst B": {"activity": 0.9, "surface_area": 150},
        },
    },
}

# User inputs
st.sidebar.header("Reaction Conditions")
temperature = st.sidebar.slider("Temperature (K)", 300, 1000, 600)
pressure = st.sidebar.slider("Pressure (bar)", 1, 100, 10)
feed_composition = st.sidebar.text_input("Feed Composition (e.g., C20H42:1.0)", "C20H42:1.0")
reactor_type = st.sidebar.selectbox("Reactor Type", ["Fixed Bed", "Fluidized Bed", "CSTR"])
catalyst_type = st.sidebar.selectbox("Catalyst Type", ["Catalyst A", "Catalyst B"])
reaction_type = st.sidebar.selectbox("Reaction Type", list(reaction_library.keys()))

# WHSV/LHSV inputs
space_velocity_type = st.sidebar.selectbox("Space Velocity Type", ["WHSV", "LHSV"])
space_velocity_value = st.sidebar.number_input(f"{space_velocity_type} (h⁻¹)", 0.1, 10.0, 1.0)

# Catalyst mass or volume (required for WHSV/LHSV calculation)
if space_velocity_type == "WHSV":
    catalyst_mass = st.sidebar.number_input("Catalyst Mass (kg)", 0.1, 100.0, 10.0)
elif space_velocity_type == "LHSV":
    catalyst_volume = st.sidebar.number_input("Catalyst Volume (m³)", 0.1, 100.0, 1.0)

# Function to get catalyst effects
def get_catalyst_effects(catalyst_type, reaction):
    catalyst_data = reaction["catalyst_effects"][catalyst_type]
    return catalyst_data["activity"], catalyst_data["surface_area"]

# Function for fixed bed reactor kinetics
def fixed_bed_kinetics(t, y, k, catalyst_activity):
    reactants = y[:len(reaction["reactants"])]
    products = y[len(reaction["reactants"]):]
    rate = k * catalyst_activity * reactants[0]  # Simplified rate equation
    dreactants_dt = [-rate]
    dproducts_dt = [rate] * len(reaction["products"])
    return dreactants_dt + dproducts_dt

# Function for CSTR kinetics
def cstr_kinetics(t, y, k, catalyst_activity, residence_time):
    reactants = y[:len(reaction["reactants"])]
    products = y[len(reaction["reactants"]):]
    rate = k * catalyst_activity * reactants[0]
    dreactants_dt = [(1.0 - reactants[0]) / residence_time - rate]  # Feed composition = 1.0
    dproducts_dt = [-products[i] / residence_time + rate for i in range(len(reaction["products"]))]
    return dreactants_dt + dproducts_dt

# Main simulation logic
if st.sidebar.button("Run Simulation"):
    # Get reaction and catalyst data
    reaction = reaction_library[reaction_type]
    activity, surface_area = get_catalyst_effects(catalyst_type, reaction)
    adjusted_rate_constant = reaction["rate_constant"] * activity

    # Calculate residence time based on WHSV or LHSV
    if space_velocity_type == "WHSV":
        residence_time = 1.0 / space_velocity_value  # Residence time = 1 / WHSV
    elif space_velocity_type == "LHSV":
        residence_time = 1.0 / space_velocity_value  # Residence time = 1 / LHSV

    # Initial conditions
    y0 = [1.0] + [0.0] * len(reaction["products"])  # [Reactant, Products]
    t_span = (0, 10)  # Simulation time
    t_eval = np.linspace(0, 10, 100)  # Time points for evaluation

    # Solve ODEs based on reactor type
    if reactor_type == "Fixed Bed":
        sol = solve_ivp(fixed_bed_kinetics, t_span, y0, args=(adjusted_rate_constant, activity), t_eval=t_eval)
    elif reactor_type == "CSTR":
        sol = solve_ivp(cstr_kinetics, t_span, y0, args=(adjusted_rate_constant, activity, residence_time), t_eval=t_eval)
    else:
        st.error("Fluidized Bed reactor model not implemented yet.")

    # Plot results
    st.header("Simulation Results")
    fig, ax = plt.subplots()
    for i, species in enumerate(reaction["reactants"] + reaction["products"]):
        ax.plot(sol.t, sol.y[i], label=species)
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.legend()
    st.pyplot(fig)

    # Display data in a table
    st.header("Simulation Data")
    data = pd.DataFrame({
        "Time": sol.t,
        **{species: sol.y[i] for i, species in enumerate(reaction["reactants"] + reaction["products"])},
    })
    st.dataframe(data)

    # Export data as CSV
    st.download_button(
        label="Download Data as CSV",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="simulation_results.csv",
        mime="text/csv",
    )