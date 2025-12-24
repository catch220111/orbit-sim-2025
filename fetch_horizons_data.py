import numpy as np
import json
from astroquery.jplhorizons import Horizons
from datetime import datetime

# Standard Mass parameters (kg)
# We store them here so we don't have to parse NASA's messy text headers.
MASS_DICT = {
    "Sun": 1.989e30,
    "Mercury": 3.3011e23,
    "Venus": 4.8675e24,
    "Earth": 5.9723e24,
    "Mars": 6.4171e23,
    "Jupiter": 1.8982e27,
    "Saturn": 5.6834e26,
    "Uranus": 8.6810e25,
    "Neptune": 1.0241e26,
    "Pluto": 1.30900e22
}

# Added 'Moon' (301) to the standard list
object_ids = {
    "Mercury": "199", 
    "Venus": "299", 
    "Earth": "399", 
    "Mars": "499",
    "Jupiter": "599", 
    "Saturn": "699", 
    "Uranus": "799", 
    "Neptune": "899"
}

def main():
    print(f"Current objects: {list(object_ids.keys())}")
    
    # 1. Allow user to add custom objects
    new_name = input("Enter new object name (or press Enter to skip): ")
    if new_name:
        new_id = input(f"Enter JPL ID for {new_name}: ")
        object_ids[new_name] = new_id
        # If it's a new object, we might need its mass
        if new_name not in MASS_DICT:
            try:
                m_val = float(input(f"Enter Mass for {new_name} (kg): "))
                MASS_DICT[new_name] = m_val
            except:
                print("Invalid mass. Defaulting to 0.")
                MASS_DICT[new_name] = 0.0

    solar_system_data = {
        "metadata":{
            "timestamp" : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "units" : {"pos":"Au", "vel":"Au/day", "mass":"kg"},
            "frame" : "Heliocentric/Ecliptic"
        },
        "bodies":[]
    }

    # 2. MANUALLY ADD THE SUN
    # Since we query location='@sun', the Sun is at (0,0,0) with 0 velocity.
    # We must include it so the N-Body loop sees it.
    print("Adding the Sun...")
    solar_system_data["bodies"].append({
        "name": "Sun",
        "id": "10",
        "position": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "mass": MASS_DICT["Sun"]
    })

    # 3. Fetch Planets
    print("Fetching planetary data from NASA JPL Horizons...")
    for name, jpl_id in object_ids.items():
        print(f"Querying {name}...")
        
        # 'vectors' gives state (x,y,z,vx,vy,vz)
        obj = Horizons(id=jpl_id, location='@sun')
        vectors = obj.vectors()

        pos = np.array([vectors['x'][0], vectors['y'][0], vectors['z'][0]]).tolist()
        vel = np.array([vectors['vx'][0], vectors['vy'][0], vectors['vz'][0]]).tolist()
        
        # Lookup mass from our dictionary
        mass = MASS_DICT.get(name, 0.0)

        planet_dict = {
            "name" : name,
            "id" : jpl_id,
            "position" : pos,
            "velocity" : vel,
            "mass": mass  # <--- Now included in the JSON
        }

        solar_system_data["bodies"].append(planet_dict)
        print(f" -> Fetched {name} (Mass: {mass:.2e} kg)")

    # 4. Save to JSON
    with open("objects.json", 'w') as f:
        json.dump(solar_system_data, f, indent=4)
    print("\nFile 'objects.json' has been successfully created with physics data.")

if __name__ == "__main__":
    main()