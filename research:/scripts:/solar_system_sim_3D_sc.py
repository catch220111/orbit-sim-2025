import json
import numpy as np
import matplotlib.pyplot as plt
import snap_hor_solarsystem as snap
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import copy
import math  # Needed for ceil
from datetime import datetime, timedelta

### ==========Constants========== ###
AU_TO_METERS = 149597870700
DAY_TO_SECONDS = 86400
G = 6.67e-11
M_sun = 1.989e30
GM = G * M_sun

### ==========State========== ###
is_paused = False
total_second_elapsed = 0
focus_index = -1 
spacecraft_list = [] 
dt = 20000 
current_zoom = 1.0 

# PHYSICS SETTINGS
# The engine will never take a step larger than this (in seconds).
# Lower = More precise orbits, Higher = Faster performance
MAX_PHYSICS_DT = 200 

PLANET_COLORS = {
    "Sun": "#FFCC33", "Mercury": "#999999", "Venus": "#FFFFCC",
    "Earth": "#3399FF", "Mars": "#CC6666", "Jupiter": "#C3997F",
    "Saturn": "#C3CFCF", "Uranus": "#ACE5EE", "Neptune": "#4169E1", "Pluto": "#AAAAAA"
}

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
        mass_kg = body.get('mass', 0.0)
        
        physics_bodies.append({
            "name": body['name'], "pos": pos_m, "vel": vel_ms, "mass": mass_kg
        })
    return physics_bodies, start_time_obj

ready_planets, start_time = load_physics_data()
initial_state = copy.deepcopy(ready_planets) 

### ==========Matplotlib Setup========== ###
fig = plt.figure(figsize=(16, 9), facecolor='black')

# 1. LAYOUT CHANGE: Main Axes moved to center with margins
# [left, bottom, width, height]
ax = fig.add_axes([0.15, 0.05, 0.70, 0.90], projection='3d')
ax.set_facecolor('black') 
ax.set_axis_off() 

# --- Grid Setup ---
grid_max = 20e11; grid_step = 1e11 
grid_floor_lines = []; grid_wall_lines = []; grid_labels = []

for x in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([x, x], [-grid_max, grid_max], [0], color='#333333', linewidth=0.6)
    grid_floor_lines.append(l)
for y in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([-grid_max, grid_max], [y, y], [0], color='#333333', linewidth=0.6)
    grid_floor_lines.append(l)

vert_alpha = 0.35; vert_color = '#555555'
for dim in ['x', 'y']:
    if dim == 'x': 
        for y in np.arange(-grid_max, grid_max + grid_step, grid_step):
            l, = ax.plot([0, 0], [y, y], [-grid_max, grid_max], color=vert_color, alpha=vert_alpha, lw=0.5, ls=':')
            grid_wall_lines.append(l)
        for z in np.arange(-grid_max, grid_max + grid_step, grid_step):
            l, = ax.plot([0, 0], [-grid_max, grid_max], [z, z], color=vert_color, alpha=vert_alpha, lw=0.5, ls=':')
            grid_wall_lines.append(l)
    else: 
        for x in np.arange(-grid_max, grid_max + grid_step, grid_step):
            l, = ax.plot([x, x], [0, 0], [-grid_max, grid_max], color=vert_color, alpha=vert_alpha, lw=0.5, ls=':')
            grid_wall_lines.append(l)
        for z in np.arange(-grid_max, grid_max + grid_step, grid_step):
            l, = ax.plot([-grid_max, grid_max], [0, 0], [z, z], color=vert_color, alpha=vert_alpha, lw=0.5, ls=':')
            grid_wall_lines.append(l)

ax.plot([-grid_max, grid_max], [0, 0], [0, 0], color='#888888', alpha=0.9, lw=1.2)
ax.plot([0, 0], [-grid_max, grid_max], [0, 0], color='#888888', alpha=0.9, lw=1.2)
ax.plot([0, 0], [0, 0], [-grid_max, grid_max], color='#888888', alpha=0.9, lw=1.2)

label_step = grid_step * 2 
for i in np.arange(-grid_max, grid_max + label_step, label_step):
    if i == 0: continue 
    val_au = i / AU_TO_METERS
    txt = f"{val_au:.1f} AU"
    grid_labels.extend([
        ax.text(i, 0, 0, txt, color='#666666', fontsize=8, ha='center', va='bottom'),
        ax.text(0, i, 0, txt, color='#666666', fontsize=8, ha='center', va='bottom'),
        ax.text(0, 0, i, txt, color='#666666', fontsize=8, ha='center', va='center')
    ])

# Initial View
base_limit = 5.5e11
ax.set_xlim(-base_limit * 2.2, base_limit * 2.2)
ax.set_ylim(-base_limit, base_limit)
ax.set_zlim(-2e11, 2e11)
ax.set_box_aspect((base_limit*2.2, base_limit, 2e11))

# --- Visuals Initialization ---
visuals = []
planet_names = [p['name'] for p in ready_planets]

for i in range(len(ready_planets)):
    p_name = ready_planets[i]['name']
    c = PLANET_COLORS.get(p_name, "#FFFFFF")
    
    if p_name == 'Sun':
        dot, = ax.plot([], [], [], 'o', color=c, ms=0, visible=False)
        trail, = ax.plot([], [], [], '-', color=c, visible=False)
    else:
        dot, = ax.plot([], [], [], 'o', color=c, ms=6, label=p_name)
        trail, = ax.plot([], [], [], '-', color=c, alpha=0.3, linewidth=1)
        
    visuals.append({
        'dot': dot, 'trail': trail,
        'x_history': deque(maxlen=400), 'y_history': deque(maxlen=400), 'z_history': deque(maxlen=400)
    })

sun_dot, = ax.plot([0], [0], [0], 'o', color="#FFCC33", ms=10, mec="#FC9601", label="Sun")
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='white',
                      fontsize=12, fontweight='bold', family='monospace')
focus_text = ax.text2D(0.02, 0.90, 'Focus: Sun', transform=ax.transAxes, color='cyan',
                       fontsize=10, fontweight='bold', family='monospace')

### ==========DASHBOARD========== ###

# 2. LAYOUT CHANGE: Moved Toggles to Left Side
ax_check = plt.axes([0.02, 0.15, 0.08, 0.1], facecolor='#333333')
check = CheckButtons(ax_check, ['Floor', 'Walls', 'Labels'], [True, True, True])
for t in check.labels: t.set_color('white'); t.set_fontsize(8); t.set_fontweight('bold')
def toggle_grid(label):
    if label == 'Floor': [l.set_visible(not l.get_visible()) for l in grid_floor_lines]
    elif label == 'Walls': [l.set_visible(not l.get_visible()) for l in grid_wall_lines]
    elif label == 'Labels': [t.set_visible(not t.get_visible()) for t in grid_labels]
    fig.canvas.draw_idle()
check.on_clicked(toggle_grid)

# --- PLANET BUTTONS ---
button_axes = []
button_objs = []
planet_buttons_names = [name for name in planet_names if name != 'Sun']
labels = ['TOP', 'ZOOM', 'Sun'] + planet_buttons_names

# 3. LAYOUT CHANGE: Moved Focus Buttons to Left Sidebar (Vertical Stack)
start_x = 0.02
start_y = 0.85
btn_w = 0.08
btn_h = 0.03
gap = 0.005

def make_callback(label):
    def on_click(event):
        global focus_index, current_zoom
        
        if label == 'TOP':
            ax.view_init(elev=90, azim=-90)
        elif label == 'ZOOM':
            current_zoom = 0.05 
        elif label == 'Sun':
            focus_index = -1
            current_zoom = 1.0
            focus_text.set_text("Focus: Sun"); focus_text.set_color('cyan')
        else:
            try:
                idx = next(i for i, p in enumerate(ready_planets) if p['name'] == label)
                focus_index = idx
                focus_text.set_text(f"Focus: {label}"); focus_text.set_color('white')
            except StopIteration:
                pass
        
        if label not in ['TOP', 'ZOOM']:
            for v in visuals: v['x_history'].clear(); v['y_history'].clear(); v['z_history'].clear()
            for s in spacecraft_list: s['history'].clear()
        
        fig.canvas.draw_idle()
    return on_click

# Generating Vertical Buttons on Left
for i, label in enumerate(labels):
    # Vertical logic instead of Grid
    y = start_y - (i * (btn_h + gap))
    
    b_ax = plt.axes([start_x, y, btn_w, btn_h])
    c = '#444444' if label in ['TOP', 'ZOOM'] else '#222222'
    btn = Button(b_ax, label, color=c, hovercolor='#111111')
    btn.label.set_color('white'); btn.label.set_fontsize(8)
    btn.on_clicked(make_callback(label))
    button_objs.append(btn); button_axes.append(b_ax) 

# Sliders
# 4. LAYOUT CHANGE: Moved Sliders to Bottom Center
ax_slider = plt.axes([0.3, 0.05, 0.4, 0.02], facecolor='#333333')
t_slider = Slider(ax_slider, 'Day', 0, 1000, valinit=0, valstep=1, color='#00FFFF')
t_slider.label.set_color('white'); t_slider.label.set_fontsize(8)

ax_speed = plt.axes([0.3, 0.02, 0.4, 0.02], facecolor='#333333')
s_slider = Slider(ax_speed, 'Warp', 1, 86400, valinit=dt, color='#00FFFF')
s_slider.label.set_color('white'); s_slider.label.set_fontsize(8)

# Launch Controls
text_boxes = []; dv_values = [0.0, 0.0, 0.0]
labels_dv = ['dVx', 'dVy', 'dVz']

# 5. LAYOUT CHANGE: Moved Launch Controls to Top Right
start_x_dv = 0.90
start_y_dv = 0.85

def submit_dv(text, idx):
    try: dv_values[idx] = float(text)
    except: pass

for i, label in enumerate(labels_dv):
    ax_box = plt.axes([start_x_dv, start_y_dv - (i * 0.04), 0.06, 0.025])
    tb = TextBox(ax_box, label, initial="0", color='#222222', hovercolor='#444444')
    tb.label.set_color('white'); tb.text_disp.set_color('white')
    tb.on_submit(lambda text, idx=i: submit_dv(text, idx))
    text_boxes.append(tb)

# Launch Button
ax_fire = plt.axes([0.895, 0.70, 0.07, 0.05])
btn_fire = Button(ax_fire, 'LAUNCH', color='#CC3333', hovercolor='#FF4444')
btn_fire.label.set_color('white'); btn_fire.label.set_weight('bold')

def fire_spacecraft(event):
    earth = next((p for p in ready_planets if p['name'] == 'Earth'), None)
    if not earth: return
    
    start_pos = earth['pos'].copy() + np.array([5e8, 0, 0]) 
    dv_vec = np.array(dv_values) * 1000.0 
    start_vel = earth['vel'].copy() + dv_vec
    
    dot, = ax.plot([], [], [], '*', color='white', ms=8, label='Ship')
    trail, = ax.plot([], [], [], '-', color='white', linewidth=1, alpha=0.5)
    
    spacecraft_list.append({
        'pos': start_pos, 'vel': start_vel,
        'dot': dot, 'trail': trail,
        'history': deque(maxlen=500)
    })
    print(f"Launched from Earth! dV: {dv_values} km/s")

btn_fire.on_clicked(fire_spacecraft)

### ==========LOGIC========== ###

def slider_update(val):
    target_day = int(val)
    global total_second_elapsed, ready_planets
    total_second_elapsed = target_day * DAY_TO_SECONDS
    current_state = copy.deepcopy(initial_state)
    sim_dt = 3600 
    steps = (target_day * DAY_TO_SECONDS) // sim_dt
    
    for _ in range(steps):
        for planet in current_state:
            pos = planet['pos']; vel = planet['vel']
            r_mag = np.linalg.norm(pos)
            accel = -GM / (r_mag**3) * pos
            vel += accel * sim_dt
            pos += vel * sim_dt
            planet['pos'] = pos; planet['vel'] = vel
            
    for i, p in enumerate(current_state):
        ready_planets[i]['pos'] = p['pos']
        ready_planets[i]['vel'] = p['vel']
        visuals[i]['x_history'].clear(); visuals[i]['y_history'].clear(); visuals[i]['z_history'].clear()
    
    spacecraft_list.clear()
    time_text.set_text(format_time(total_second_elapsed))
    fig.canvas.draw_idle()

t_slider.on_changed(slider_update) 

def speed_update(val):
    global dt; dt = val
s_slider.on_changed(speed_update)

def on_press(event):
    global is_paused, current_zoom
    if event.key == 'p' or event.key == 'P': 
        is_paused = not is_paused
    if event.key == '=' or event.key == '+':
        current_zoom *= 0.8
    if event.key == '-' or event.key == '_':
        current_zoom *= 1.2
        
fig.canvas.mpl_connect('key_press_event', on_press)

def update(frame):
    global total_second_elapsed, dt, current_zoom
    if is_paused: return 
    
    # --- SUB-STEPPING LOGIC ---
    # We slice the huge 'dt' into smaller 'sub_dt' chunks
    remaining_dt = dt
    
    while remaining_dt > 0:
        # Take a step, but don't exceed MAX_PHYSICS_DT
        step = min(remaining_dt, MAX_PHYSICS_DT)
        
        # 1. Update Planets (Micro-step)
        for i in range(len(ready_planets)):
            planet = ready_planets[i]
            pos = planet['pos']; vel = planet['vel']
            r_mag = np.linalg.norm(pos)
            accel = -GM / (r_mag ** 3) * pos
            vel += accel * step
            pos += vel * step
            planet['pos'] = pos; planet['vel'] = vel

        # 2. Update Spacecraft (Micro-step)
        for ship in spacecraft_list:
            pos = ship['pos']; vel = ship['vel']
            r_sun = -pos; d_sun = np.linalg.norm(r_sun)
            accel_total = (G * M_sun / d_sun**3) * r_sun
            
            for planet in ready_planets:
                r_vec = planet['pos'] - pos
                dist = np.linalg.norm(r_vec)
                # Collision/Singularity Guard
                if dist > 1e6: 
                    accel_total += (G * planet['mass'] / dist**3) * r_vec
            
            vel += accel_total * step
            pos += vel * step
            ship['pos'] = pos; ship['vel'] = vel
            
        remaining_dt -= step
        
    # --- DRAWING (Only once per frame) ---
    total_second_elapsed += dt
    time_text.set_text(format_time(total_second_elapsed))

    if focus_index == -1: offset = np.array([0.0, 0.0, 0.0])
    else: offset = ready_planets[focus_index]['pos'].copy()
    
    sun_disp = -offset; sun_dot.set_data([sun_disp[0]], [sun_disp[1]]); sun_dot.set_3d_properties([sun_disp[2]])
    
    l_y = 5.5e11 * current_zoom
    l_x = l_y * 2.2; l_z = 2e11 * current_zoom
    ax.set_xlim(-l_x, l_x); ax.set_ylim(-l_y, l_y); ax.set_zlim(-l_z, l_z)

    # Update Artists
    for i in range(len(ready_planets)):
        artist = visuals[i]; draw_pos = ready_planets[i]['pos'] - offset 
        artist['x_history'].append(draw_pos[0]); artist['y_history'].append(draw_pos[1]); artist['z_history'].append(draw_pos[2])
        artist['dot'].set_data([draw_pos[0]], [draw_pos[1]]); artist['dot'].set_3d_properties([draw_pos[2]])
        artist['trail'].set_data(list(artist['x_history']), list(artist['y_history'])); artist['trail'].set_3d_properties(list(artist['z_history']))

    for ship in spacecraft_list:
        draw_pos = ship['pos'] - offset
        ship['history'].append(draw_pos)
        hist = np.array(ship['history'])
        if len(hist) > 0:
            ship['dot'].set_data([draw_pos[0]], [draw_pos[1]]); ship['dot'].set_3d_properties([draw_pos[2]])
            ship['trail'].set_data(hist[:,0], hist[:,1]); ship['trail'].set_3d_properties(hist[:,2])
    return 

def format_time(seconds):
    current_date = start_time + timedelta(seconds=seconds)
    return current_date.strftime('Mission Time: %Y-%m-%d %H:%M')

try:
    mng = plt.get_current_fig_manager()
    if hasattr(mng, 'resize') and hasattr(mng, 'window'): mng.resize(*mng.window.maxsize())
    elif hasattr(mng, 'full_screen_toggle'): mng.full_screen_toggle()
except: pass

ani = FuncAnimation(fig, update, frames=None, interval=20, blit=False, cache_frame_data=False)
plt.show()