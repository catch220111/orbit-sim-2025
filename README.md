# 🚀 interplanetary-mission-sim: JPL Horizons N-Body Suite

A high-fidelity orbital mechanics environment that bridges real-world NASA ephemeris data with a 3D N-body gravitational engine. This suite allows for the planning and execution of interplanetary transfers, featuring a functional "Flight Computer" for trajectory correction and orbital insertion.

## 🛠 Project Architecture

The project is split into two specialized modules to maintain professional data separation:

1. **`fetch_horizons_data.py` (The Data Fetcher):** Uses the `astroquery.jplhorizons` library to pull precise position and velocity vectors directly from NASA's JPL Horizons system. It exports this state to `objects.json`.
    
2. **`main.py` (The Simulation Engine):** A 3D Matplotlib-based engine that simulates gravitational interactions using Newton's Law of Universal Gravitation and provides a real-time HUD for mission control.
    

## 🔬 Physics & Engineering Features

- **N-Body Gravity:** Unlike simple 2-body models, this sim calculates the gravitational pull from the Sun and all active planets on the spacecraft simultaneously.
    
- **NASA-Grade Flight Computer:** * **Shooting Method:** The "FIX PATH" feature uses iterative simulation to calculate the necessary $\Delta V$ (velocity change) to intercept Mars.
    
    - **B-Plane Targeting:** Calculates a safety offset (6,000 km radius) to ensure the spacecraft achieves a safe flyby altitude rather than a direct impact.
        
- **Variable Precision Integrator:** Uses a dynamic time-stepping logic that increases calculation frequency (precision) when a spacecraft is in close proximity to a planetary body.
    

## 🎮 Mission Control UI

- **Navigation:** Focused view modes for all major planets and "Top-Down" ecliptic perspectives.
    
- **Propulsion:** Manual Prograde/Retrograde burn controls and a custom $\Delta V$ vector launch panel.
    
- **Telemetry HUD:** Real-time display of altitude, relative velocity, and estimated periapsis during planetary approach.
    

## 🚀 Getting Started

### 1. Requirements

Bash

```
pip install numpy matplotlib astroquery
```

### 2. Update Data (Optional)

Run the fetcher to get the latest planetary positions from NASA:

Bash

```
python fetch_horizons_data.py
```

### 3. Run Simulation

Bash

```
python main.py
```

## 📈 Roadmap for Professional Alignment

- [ ] **Integrator Upgrade:** Move from Euler-based stepping to a **Runge-Kutta (RK4)** integrator for long-term orbital stability.
    
- [ ] **Non-Spherical Gravity:** Implement **J2 Perturbations** for high-accuracy Low Earth Orbit (LEO) modeling.
    
- [ ] **Relativistic Effects:** Add Schwarzschild metric corrections for high-precision Mercury orbits.