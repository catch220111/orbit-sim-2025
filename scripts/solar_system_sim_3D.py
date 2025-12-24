import json
import numpy as np
import matplotlib.pyplot as plt
import snap_hor_solarsystem as snap
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
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
focus_index = -1 
planet_buttons = []

### ==========Matplotlib Setup========== ###
# 1. Setup the Figure (Widescreen 16:9)
# NEW: facecolor='black' makes the entire window wallpaper black
fig = plt.figure(figsize=(16, 9), facecolor='black')

# 2. Main 3D Plot - Occupies top 92% of the screen
ax = fig.add_axes([0.0, 0.08, 1.0, 0.92], projection='3d')
ax.set_facecolor('black') # Match the figure background

# --- TRUE INFINITE SPACE SETUP ---
ax.set_axis_off() 

# Lists to store grid parts (so we can toggle them later)
grid_floor_lines = []
grid_wall_lines = []
grid_labels = []

grid_max = 20e11  
grid_step = 1e11 

# --- A. THE MAIN FLOOR (XY Plane) ---
for x in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([x, x], [-grid_max, grid_max], [0], color='#333333', linewidth=0.6)
    grid_floor_lines.append(l)
for y in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([-grid_max, grid_max], [y, y], [0], color='#333333', linewidth=0.6)
    grid_floor_lines.append(l)

# --- B. THE VERTICAL WALLS (XZ and YZ Planes) ---
vert_alpha = 0.35
vert_color = '#555555'

for y in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([0, 0], [y, y], [-grid_max, grid_max], color=vert_color, alpha=vert_alpha, linewidth=0.5, linestyle=':')
    grid_wall_lines.append(l)
for z in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([0, 0], [-grid_max, grid_max], [z, z], color=vert_color, alpha=vert_alpha, linewidth=0.5, linestyle=':')
    grid_wall_lines.append(l)

for x in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([x, x], [0, 0], [-grid_max, grid_max], color=vert_color, alpha=vert_alpha, linewidth=0.5, linestyle=':')
    grid_wall_lines.append(l)
for z in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([-grid_max, grid_max], [0, 0], [z, z], color=vert_color, alpha=vert_alpha, linewidth=0.5, linestyle=':')
    grid_wall_lines.append(l)

# --- C. CENTER AXES (Crosshairs) ---
ax.plot([-grid_max, grid_max], [0, 0], [0, 0], color='#888888', alpha=0.9, linewidth=1.2)
ax.plot([0, 0], [-grid_max, grid_max], [0, 0], color='#888888', alpha=0.9, linewidth=1.2)
ax.plot([0, 0], [0, 0], [-grid_max, grid_max], color='#888888', alpha=0.9, linewidth=1.2)

# --- D. READINGS (Text Labels) ---
label_step = grid_step * 2 
for i in np.arange(-grid_max, grid_max + label_step, label_step):
    if i == 0: continue 
    
    val_au = i / AU_TO_METERS
    label_text = f"{val_au:.1f} AU"
    
    t1 = ax.text(i, 0, 0, label_text, color='#666666', fontsize=8, ha='center', va='bottom')
    t2 = ax.text(0, i, 0, label_text, color='#666666', fontsize=8, ha='center', va='bottom')
    t3 = ax.text(0, 0, i, label_text, color='#666666', fontsize=8, ha='center', va='center')
    
    grid_labels.extend([t1, t2, t3])


# --- View Limits & Aspect Ratio ---
limit_y = 5.5e11
limit_x = limit_y * 2.2 
limit_z = 2e11

ax.set_xlim(-limit_x, limit_x)
ax.set_ylim(-limit_y, limit_y)
ax.set_zlim(-limit_z, limit_z)

ax.set_box_aspect((limit_x, limit_y, limit_z))

# --- Visuals Initialization ---
planet_names = list(snap.object_ids.keys())
visuals = []
number_of_body = len(snap.object_ids)
colors = ["#999999", "#FFFFCC", "#3399FF", "#CC6666", "#C3997F", "#C3CFCF", "#ACE5EE", "#4169E1"]

for i in range(number_of_body):
    dot, = ax.plot([], [], [], 'o', color=colors[i], ms=6, label=planet_names[i])
    trail, = ax.plot([], [], [], '-', color=colors[i], alpha=0.3, linewidth=1)
    
    visuals.append({
        'dot': dot,
        'trail': trail,
        'x_history': deque(maxlen=400),
        'y_history': deque(maxlen=400),
        'z_history': deque(maxlen=400)
    })

# --- HUD (Inside the Plot) ---
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white',
                      fontsize=12, fontweight='bold', family='monospace',
                      bbox=dict(facecolor='black', edgecolor='white', alpha=0.7))

focus_text = ax.text2D(0.02, 0.90, 'Focus: Sun', transform=ax.transAxes, color='cyan',
                       fontsize=10, fontweight='bold', family='monospace')

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
        physics_bodies.append({"name": body['name'], "pos": pos_m, "vel": vel_ms})
    return physics_bodies, start_time_obj

ready_planets, start_time = load_physics_data()
sun_dot, = ax.plot([0], [0], [0], 'o', color="#FFCC33", ms=10, mec="#FC9601", label="Sun")


### ==========DASHBOARD CONTROLS========== ###

# 1. Background Panel (Gunmetal Grey for Contrast)
# This sits ON TOP of the black wallpaper
dashboard_bg = plt.axes([0.0, 0.0, 1.0, 0.08], facecolor='#333333')
dashboard_bg.set_xticks([]); dashboard_bg.set_yticks([])

# 2. Toggle Switches (Left Corner)
ax_check = plt.axes([0.01, 0.01, 0.08, 0.06], facecolor='#333333')
check = CheckButtons(ax_check, ['Floor', 'Walls', 'Labels'], [True, True, True])

# Standard Styling
for t in check.labels:
    t.set_color('white')
    t.set_fontsize(9)
    t.set_fontweight('bold')

def toggle_grid(label):
    if label == 'Floor':
        for line in grid_floor_lines: line.set_visible(not line.get_visible())
    elif label == 'Walls':
        for line in grid_wall_lines: line.set_visible(not line.get_visible())
    elif label == 'Labels':
        for text in grid_labels: text.set_visible(not text.get_visible())
    fig.canvas.draw_idle()

check.on_clicked(toggle_grid)

# 3. Planet Selector Buttons (Middle Left)
button_axes = []
button_objs = []
labels = ['Sun'] + planet_names

start_x = 0.12
btn_w = 0.06; btn_h = 0.03; gap = 0.005

def make_callback(label):
    def on_click(event):
        global focus_index
        if label == 'TOP':
            ax.view_init(elev=90, azim=-90)
            print("Camera set to Top-Down")
        elif label == 'ZOOM':
            # 0.1 AU is a good distance to see a planet clearly
            zoom_limit = 0.1 * AU_TO_METERS 
            ax.set_xlim(-zoom_limit * 2.2, zoom_limit * 2.2) # Keep widescreen ratio
            ax.set_ylim(-zoom_limit, zoom_limit)
            ax.set_zlim(-zoom_limit * 0.5, zoom_limit * 0.5)
            print("Zoomed into Focus")
        elif label == 'Sun':
            focus_index = -1
            focus_text.set_text("Focus: Sun")
            # Reset to default wide view
            limit_y = 5.5e11
            ax.set_xlim(-limit_y * 2.2, limit_y * 2.2)
            ax.set_ylim(-limit_y, limit_y)
            ax.set_zlim(-2e11, 2e11)
        else:
            focus_index = planet_names.index(label)
            focus_text.set_text(f"Focus: {label}")
            # Optional: Auto-zoom when selecting a planet? 
            # (Uncomment the line below if you want that)
            # on_click(type('obj', (object,), {'label': 'ZOOM'}))
            
        if label not in ['TOP', 'ZOOM']:
            for v in visuals:
                v['x_history'].clear(); v['y_history'].clear(); v['z_history'].clear()
        
        fig.canvas.draw_idle()
    return on_click


# Updated labels list
labels = ['TOP', 'ZOOM', 'Sun'] + planet_names

# Increase grid columns to 7
for i, label in enumerate(labels):
    col = i % 7  # Changed to 7
    row = 1 if i < 7 else 0
    
    x = start_x + col * (btn_w + gap)
    y = 0.045 if row == 1 else 0.01 
    
    b_ax = plt.axes([x, y, btn_w, btn_h])
    
    # Style TOP and ZOOM buttons to look like a "Navigation Group"
    btn_color = '#444444' if label in ['TOP', 'ZOOM'] else '#222222'
    btn = Button(b_ax, label, color=btn_color, hovercolor='#111111')
    
    btn.label.set_color('white'); btn.label.set_fontsize(8)
    btn.on_clicked(make_callback(label))
    button_axes.append(b_ax); button_objs.append(btn)

# 4. Sliders (Right Side)
ax_slider = plt.axes([0.55, 0.045, 0.4, 0.02], facecolor='#333333')
t_slider = Slider(ax_slider, 'Day', 0, 1000, valinit=0, valstep=1, color='#00FFFF')
t_slider.label.set_color('white'); t_slider.label.set_fontsize(8)

ax_speed = plt.axes([0.55, 0.01, 0.4, 0.02], facecolor='#333333')
s_slider = Slider(ax_speed, 'Warp', 1, 86400, valinit=dt, color='#00FFFF')
s_slider.label.set_color('white'); s_slider.label.set_fontsize(8)


### ==========Logic========== ###
initial_state = copy.deepcopy(ready_planets)

def format_time(seconds):
    current_date = start_time + timedelta(seconds=seconds)
    return current_date.strftime('Mission Time: %Y-%m-%d %H:%M')

def slider_update(val):
    target_day = int(val)
    global total_second_elapsed
    total_second_elapsed = target_day * DAY_TO_SECONDS
    current_state = copy.deepcopy(initial_state)

    for _ in range(target_day):
        for planet in current_state:
            r_vec = planet['pos'] 
            r_mag = np.linalg.norm(r_vec)
            accel = -GM * r_vec / r_mag**3
            planet['vel'] += accel * DAY_TO_SECONDS
            planet['pos'] += planet['vel'] * DAY_TO_SECONDS

    if focus_index == -1: offset = np.array([0.0, 0.0, 0.0])
    else: offset = current_state[focus_index]['pos'].copy()

    sun_disp = np.array([0.0, 0.0, 0.0]) - offset
    sun_dot.set_data([sun_disp[0]], [sun_disp[1]])
    sun_dot.set_3d_properties([sun_disp[2]])

    for i in range(len(current_state)):
        draw_pos = current_state[i]['pos'] - offset
        v = visuals[i]
        v['dot'].set_data([draw_pos[0]], [draw_pos[1]])
        v['dot'].set_3d_properties([draw_pos[2]])
        v['x_history'].clear(); v['y_history'].clear(); v['z_history'].clear()
        v['trail'].set_data([], []); v['trail'].set_3d_properties([])
        ready_planets[i]['pos'] = current_state[i]['pos']
        ready_planets[i]['vel'] = current_state[i]['vel']
        
    time_text.set_text(format_time(total_second_elapsed))
    fig.canvas.draw_idle()

def speed_update(val):
    global dt
    dt = val

t_slider.on_changed(slider_update) 
s_slider.on_changed(speed_update)

def on_press(event):
    global is_paused
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    scale = 1.2
    
    if event.key == "=" or event.key == "+":
        ax.set_xlim(xlim[0]/scale, xlim[1]/scale)
        ax.set_ylim(ylim[0]/scale, ylim[1]/scale)
        ax.set_zlim(zlim[0]/scale, zlim[1]/scale)
    elif event.key == '-' or event.key == '_':
        ax.set_xlim(xlim[0]*scale, xlim[1]*scale)
        ax.set_ylim(ylim[0]*scale, ylim[1]*scale)
        ax.set_zlim(zlim[0]*scale, zlim[1]*scale)

    if event.key == 'p' or event.key == 'P':
        is_paused = not is_paused

fig.canvas.mpl_connect('key_press_event', on_press)

def update(frame):
    global total_second_elapsed, dt
    if is_paused: return 
    total_second_elapsed += dt
    time_text.set_text(format_time(total_second_elapsed))

    if focus_index == -1: offset = np.array([0.0, 0.0, 0.0])
    else: offset = ready_planets[focus_index]['pos'].copy()

    sun_disp = np.array([0.0, 0.0, 0.0]) - offset
    sun_dot.set_data([sun_disp[0]], [sun_disp[1]])
    sun_dot.set_3d_properties([sun_disp[2]])

    for i in range(len(ready_planets)):
        planet = ready_planets[i]; artist = visuals[i]
        pos = planet['pos']; vel = planet['vel']
        r_mag = np.linalg.norm(pos)
        accel = -GM / (r_mag ** 3) * pos
        vel += accel * dt; pos += vel * dt
        planet['pos'] = pos; planet['vel'] = vel

        draw_pos = pos - offset 
        artist['x_history'].append(draw_pos[0]); artist['y_history'].append(draw_pos[1]); artist['z_history'].append(draw_pos[2])
        artist['dot'].set_data([draw_pos[0]], [draw_pos[1]])
        artist['dot'].set_3d_properties([draw_pos[2]])
        artist['trail'].set_data(list(artist['x_history']), list(artist['y_history']))
        artist['trail'].set_3d_properties(list(artist['z_history']))
    return 

# Maximize Window on Start (MacOS compatible)
try:
    mng = plt.get_current_fig_manager()
    if hasattr(mng, 'resize') and hasattr(mng, 'window'): mng.resize(*mng.window.maxsize())
    elif hasattr(mng, 'full_screen_toggle'): mng.full_screen_toggle()
except: pass

ani = FuncAnimation(fig, update, frames=None, interval=20, blit=False, cache_frame_data=False)
plt.show()