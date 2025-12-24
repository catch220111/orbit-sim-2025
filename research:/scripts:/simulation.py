import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# 1. PHYSICAL CONSTANTS (S.I. Units)
G = 6.67430e-11
M_SUN = 1.989e30
MU = G * M_SUN
AU_TO_METERS = 149597870700
DAY_TO_SECONDS = 86400

# 2. DATA LOADING
def load_and_convert_data(filename="objects.json"):
    """Reads JSON and converts AU/Day to Meters/Second."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run your snapshot script first!")
        return []

    processed_bodies = []
    for body in data['bodies']:
        # Convert units
        pos_m = np.array(body['position']) * AU_TO_METERS
        vel_ms = (np.array(body['velocity']) * AU_TO_METERS) / DAY_TO_SECONDS
        
        processed_bodies.append({
            "name": body['name'],
            "pos": pos_m,
            "vel": vel_ms
        })
    return processed_bodies

# --- INITIALIZE DATA ---
# We call this outside the function so 'planets' is available globally
planets = load_and_convert_data()

# 3. VISUAL SETUP
fig, ax = plt.subplots(figsize=(9, 9))
ax.set_aspect('equal')
ax.set_facecolor('black')
# Set limits to roughly 2 AU (Inner Solar System)
ax.set_xlim(-5.5e11, 5.5e11)
ax.set_ylim(-5.5e11, 5.5e11)

# Draw the Sun
ax.plot(0, 0, 'yo', markersize=10, label="Sun", markeredgecolor="orange")

# Containers for animation objects
drawing_objects = []
TRAIL_LENGTH = 150 # Days of history to show

for p in planets:
    line, = ax.plot([], [], '-', linewidth=1, alpha=0.5) # Trail
    dot, = ax.plot([], [], 'o', markersize=6, label=p['name']) # Planet
    drawing_objects.append({
        'line': line,
        'dot': dot,
        'x_trail': deque(maxlen=TRAIL_LENGTH),
        'y_trail': deque(maxlen=TRAIL_LENGTH)
    })

# The Clock Text
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', 
                    fontsize=12, family='monospace', fontweight='bold')

# 4. SIMULATION VARIABLES
dt = 86400  # 1 day per frame
total_seconds = 0

# 5. PHYSICS & ANIMATION LOOP
def update(frame):
    global total_seconds
    updated_artists = []
    
    # Update Time
    total_seconds += dt
    years = int(total_seconds // (365 * 86400))
    days = int((total_seconds % (365 * 86400)) // 86400)
    time_text.set_text(f'Time: {years:02d}y {days:03d}d')
    updated_artists.append(time_text)

    for i, p in enumerate(planets):
        # --- PHYSICS (Euler-Cromer) ---
        r_vec = p['pos']
        r_mag = np.linalg.norm(r_vec)
        
        # Acceleration: a = -GM / r^3 * r
        accel = -MU * r_vec / r_mag**3
        
        # Step velocity, then position
        p['vel'] += accel * dt
        p['pos'] += p['vel'] * dt
        
        # --- UPDATE VISUALS ---
        obj = drawing_objects[i]
        obj['x_trail'].append(p['pos'][0])
        obj['y_trail'].append(p['pos'][1])
        
        obj['dot'].set_data([p['pos'][0]], [p['pos'][1]])
        obj['line'].set_data(list(obj['x_trail']), list(obj['y_trail']))
        
        updated_artists.extend([obj['dot'], obj['line']])
        
    return updated_artists

# 6. RUN
# interval=20 means 50 frames per second
ani = FuncAnimation(fig, update, frames=None, interval=20, blit=True)
plt.legend(loc='upper right', fontsize='x-small')
plt.title("Inner Solar System Simulation (Heliocentric)")

### ==========Dynamic Zoon========== ###
def on_press(event):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scale_factor = 1.2

    if event.key == "=" or event.key == "+":
        ax.set_xlim(xlim[0] / scale_factor, xlim[1] / scale_factor)
        ax.set_ylim(ylim[0] / scale_factor, ylim[1] / scale_factor)

    elif event.key == '-' or event.key == '_':
        ax.set_xlim(xlim[0] * scale_factor, xlim[1] * scale_factor)
        ax.set_ylim(ylim[0] * scale_factor, ylim[1] * scale_factor)

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_press)

plt.show()