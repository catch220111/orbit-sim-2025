import json
import numpy as np
import matplotlib.pyplot as plt
import snap_hor_solarsystem as snap
from matplotlib.animation import FuncAnimation
from collections import deque
from matplotlib.widgets import Slider
import copy
from datetime import datetime, timedelta

### ==========Constants========== ###
AU_TO_METERS = 149597870700
DAY_TO_SECONDS = 86400
dt = 6
G = 6.67e-11
M_sun = 1.989e30
GM = G * M_sun


### ==========State========== ###
is_paused = False
total_second_elapsed = 0

'''
### ==========Euler Prediction========== ###
r_mag = np.linalg.norm(pos)
accel = -GM / (r_mag ** 3) * pos
vel +=  accel * dt
pos += vel * dt
'''



### ==========Matplotlib Setup========== ###
fig, ax = plt.subplots(figsize=(8,8))
ax.set_facecolor('k')
plt.xlim(-5.5e11, 5.5e11)
plt.ylim(-5.5e11, 5.5e11)
ax.set_aspect('equal')

planet_names = list(snap.object_ids.keys())
visuals = []
number_of_body = len(snap.object_ids)
colors = ["#999999", "#FFFFCC", "#3399FF", "#CC6666", "#C3997F", "#C3CFCF", "#ACE5EE", "#4169E1" ]
for i in range(number_of_body):
    dot, = ax.plot([], [], 'o', color = colors[i], ms = 6, label = planet_names[i])

    trail, = ax.plot([],[], '-', color = colors[i], alpha = 0.3, linewidth = 1)
    visuals.append({
        'dot' : dot,
        'trail' : trail,
        'x_history' : deque(maxlen = 1000),
        'y_history' : deque(maxlen = 1000)
        })
ax.legend(loc = 'upper right', fontsize = 'small')

time_text = ax.text(0.02, 0.96, '', color='white', transform=ax.transAxes, 
                    fontsize=11, fontweight='bold', family='monospace',
                    bbox=dict(facecolor='black', alpha=0.5)) # Added a small background box for readability

        
### =========Data Loading========== ###
def load_physics_data(filename="objects.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    start_time_str = data['metadata']['timestamp']
    start_time_obj = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')



    physics_bodies = []
    for body in data['bodies']:
        pos_m = np.array(body['position']) * AU_TO_METERS
        vel_ms = (np.array(body['velocity']) * AU_TO_METERS) / DAY_TO_SECONDS
        
        physics_bodies.append({
            "name": body['name'],
            "pos": pos_m,
            "vel": vel_ms
        })
    return physics_bodies, start_time_obj

ready_planets, start_time = load_physics_data()



ax.plot(0,0, 'o', color = "#FFCC33", ms = 10, mec = "#FC9601", label ="Sun")

### ==========Dynamic Zoon========== ###
def on_press(event):
    global is_paused
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scale_factor = 1.2

    if event.key == "=" or event.key == "+":
        ax.set_xlim(xlim[0] / scale_factor, xlim[1] / scale_factor)
        ax.set_ylim(ylim[0] / scale_factor, ylim[1] / scale_factor)

    elif event.key == '-' or event.key == '_':
        ax.set_xlim(xlim[0] * scale_factor, xlim[1] * scale_factor)
        ax.set_ylim(ylim[0] * scale_factor, ylim[1] * scale_factor)

    
    if event.key == 'p' or event.key == 'P':
        is_paused = not is_paused
        if is_paused:
            print("Simulation Paused")
        else:
            print("Simulation Resumed")

    fig.canvas.draw_idle()

fig.canvas.mpl_connect('key_press_event', on_press)



### ==========Slider========= ###
plt.subplots_adjust(bottom = 0.2)
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor = 'gray')

t_slider = Slider(ax_slider, 'Day', 0, 1000, valinit=0, valstep=1)

initial_state = copy.deepcopy(ready_planets)

def slider_update(val):
    # Fix 1: Correctly get the integer value from the slider
    target_day = int(val)
    global total_second_elapsed
    total_second_elapsed = target_day * DAY_TO_SECONDS
    print(f"Time Machine: Jumping to Day {target_day}")

    # Reset to start
    current_state = copy.deepcopy(initial_state)

    # FAST-FORWARD Loop
    for _ in range(target_day):
        for planet in current_state:
            r_vec = planet['pos'] # Fix 2: Changed 'p' to 'planet'
            r_mag = np.linalg.norm(r_vec)
            
            # Basic physics step
            accel = -GM * r_vec / r_mag**3
            planet['vel'] += accel * DAY_TO_SECONDS
            planet['pos'] += planet['vel'] * DAY_TO_SECONDS

    # Update Visuals
    for i in range(len(current_state)):
        pos = current_state[i]['pos']
        visuals[i]['dot'].set_data([pos[0]], [pos[1]])
        
        # Clear trails so they don't look messy after a jump
        visuals[i]['x_history'].clear()
        visuals[i]['y_history'].clear()
        
        # Fix 3: Sync the main physics data so the animator resumes from here
        ready_planets[i]['pos'] = current_state[i]['pos']
        ready_planets[i]['vel'] = current_state[i]['vel']
    time_text.set_text(format_time(total_second_elapsed))
    fig.canvas.draw_idle()

t_slider.on_changed(slider_update) 
            

### ==========Warp========== ###
plt.subplots_adjust(bottom=0.25)
ax_speed = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor='gray')
s_slider = Slider(ax_speed, 'Warp ', 1, 86400, valinit=dt)

def speed_update(val):
    global dt
    dt = val
    print(f"Warp Speed: {dt:.1f} s/frame")

s_slider.on_changed(speed_update)


def update(frame):
    global total_second_elapsed, dt
    if is_paused:
        return [v['dot'] for v in visuals] + [v['trail'] for v in visuals] + [time_text]
    updated_visuals = []
    total_second_elapsed += dt
    time_text.set_text(format_time(total_second_elapsed))


    for i in range(len(ready_planets)):
        planet = ready_planets[i]
        artist = visuals[i]

        # 1. Math
        pos = planet['pos']
        vel = planet['vel']
        r_mag = np.linalg.norm(pos)
        accel = -GM / (r_mag ** 3) * pos
        
        vel += accel * dt
        pos += vel * dt

        # 2. Save back to dictionary
        planet['pos'] = pos
        planet['vel'] = vel

        artist['x_history'].append(pos[0])
        artist['y_history'].append(pos[1])
        # 3. Update Visuals (Pass as lists: [x], [y])
        artist['dot'].set_data([pos[0]], [pos[1]])
        artist['trail'].set_data(list(artist['x_history']), list(artist['y_history']))
        updated_visuals.extend([artist['dot'], artist['trail']])
    
    updated_visuals.append(time_text)
    return updated_visuals

### ========= Time Setup========= ###
def format_time(seconds):
    days = int(seconds // DAY_TO_SECONDS)
    remaining_seconds = seconds % DAY_TO_SECONDS
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    current_date = start_time + timedelta(seconds=seconds)
    return current_date.strftime('Date: %Y %m %d %H:%M')


ani = FuncAnimation(fig, update, frames = None, interval = 20, blit = True, cache_frame_data = False)
plt.title("Inner Solar System Simulation (Heliocentric)")
plt.show()