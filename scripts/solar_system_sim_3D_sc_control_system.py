import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import copy
from datetime import datetime, timedelta

### ==========Constants========== ###
AU_TO_METERS = 149597870700
DAY_TO_SECONDS = 86400
G = 6.67430e-11 # Higher precision G
M_sun = 1.989e30
GM = G * M_sun

### ==========State========== ###
is_paused = False
total_second_elapsed = 0
focus_index = -1 
spacecraft_list = [] 
active_ship_idx = -1 
dt = 2000 
current_zoom = 1.0 

# PHYSICS SETTINGS
# CHANGED FROM 200 TO 1. 
# This is the "High Precision" mode. It calculates gravity every 1 second.
# This eliminates the "drift" error in Python simulations.
MAX_PHYSICS_DT = 1 

PLANET_COLORS = {
    "Sun": "#FFCC33", "Mercury": "#999999", "Venus": "#FFFFCC",
    "Earth": "#3399FF", "Mars": "#CC6666", "Jupiter": "#C3997F",
    "Saturn": "#C3CFCF", "Uranus": "#ACE5EE", "Neptune": "#4169E1", "Pluto": "#AAAAAA"
}

### =========Data Loading========== ###
def load_physics_data(filename="objects.json"):
    try:
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
    except:
        return [{'name': 'Sun', 'pos': np.zeros(3), 'vel': np.zeros(3), 'mass': M_sun},
                {'name': 'Earth', 'pos': np.array([AU_TO_METERS, 0, 0]), 'vel': np.array([0, 29780, 0]), 'mass': 5.97e24}], datetime.now()

ready_planets, start_time = load_physics_data()
initial_state = copy.deepcopy(ready_planets) 

### ==========Matplotlib Setup========== ###
fig = plt.figure(figsize=(16, 9), facecolor='black')
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

base_limit = 5.5e11
ax.set_xlim(-base_limit * 2.2, base_limit * 2.2)
ax.set_ylim(-base_limit, base_limit)
ax.set_zlim(-2e11, 2e11)
ax.set_box_aspect((base_limit*2.2, base_limit, 2e11))

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

### ==========UI: SIDEBARS========== ###
ui_widgets = [] 

def add_btn(rect, label, callback, color='#333333', hover='#444444'):
    ax_btn = plt.axes(rect)
    btn = Button(ax_btn, label, color=color, hovercolor=hover)
    btn.label.set_color('white')
    btn.label.set_fontsize(8)
    btn.on_clicked(callback)
    ui_widgets.append(btn)
    return btn

# --- LEFT SIDEBAR (Controls) ---
ax_check = plt.axes([0.02, 0.15, 0.08, 0.1], facecolor='#333333')
check = CheckButtons(ax_check, ['Floor', 'Walls', 'Labels'], [True, True, True])
for t in check.labels: t.set_color('white'); t.set_fontsize(8); t.set_fontweight('bold')
def toggle_grid(label):
    if label == 'Floor': [l.set_visible(not l.get_visible()) for l in grid_floor_lines]
    elif label == 'Walls': [l.set_visible(not l.get_visible()) for l in grid_wall_lines]
    elif label == 'Labels': [t.set_visible(not t.get_visible()) for t in grid_labels]
    fig.canvas.draw_idle()
check.on_clicked(toggle_grid)

start_x = 0.02; start_y = 0.85; btn_w = 0.08; btn_h = 0.03; gap = 0.005
left_labels = ['TOP', 'ZOOM', 'Sun'] + [n for n in planet_names if n != 'Sun']

def make_callback(label):
    def on_click(event):
        global focus_index, current_zoom
        if label == 'TOP': ax.view_init(elev=90, azim=-90)
        elif label == 'ZOOM': current_zoom = 0.05 
        elif label == 'Sun':
            focus_index = -1; current_zoom = 1.0
            focus_text.set_text("Focus: Sun"); focus_text.set_color('cyan')
        else:
            try:
                idx = next(i for i, p in enumerate(ready_planets) if p['name'] == label)
                focus_index = idx
                focus_text.set_text(f"Focus: {label}"); focus_text.set_color('white')
            except: pass
        
        if label not in ['TOP', 'ZOOM']:
            for v in visuals: v['x_history'].clear(); v['y_history'].clear(); v['z_history'].clear()
            for s in spacecraft_list: s['history'].clear()
        fig.canvas.draw_idle()
    return on_click

for i, label in enumerate(left_labels):
    y = start_y - (i * (btn_h + gap))
    c = '#444444' if label in ['TOP', 'ZOOM'] else '#222222'
    add_btn([start_x, y, btn_w, btn_h], label, make_callback(label), color=c)

# --- TOP RIGHT (Launch) ---
dv_values = [0.0, 0.0, 0.0]
labels_dv = ['dVx', 'dVy', 'dVz']
start_x_dv = 0.90; start_y_dv = 0.85

def submit_dv(text, idx):
    try: dv_values[idx] = float(text)
    except: pass

for i, label in enumerate(labels_dv):
    ax_box = plt.axes([start_x_dv, start_y_dv - (i * 0.04), 0.06, 0.025])
    tb = TextBox(ax_box, label, initial="0", color='#222222', hovercolor='#444444')
    tb.label.set_color('white'); tb.text_disp.set_color('white')
    tb.on_submit(lambda text, idx=i: submit_dv(text, idx))
    ui_widgets.append(tb)

def fire_spacecraft(event):
    global active_ship_idx
    earth = next((p for p in ready_planets if p['name'] == 'Earth'), None)
    if not earth: return
    
    # 6.5e6 meters is approx 6,500 km (Earth Radius + 130km)
    # Launching from Equator (X-axis offset)
    start_pos = earth['pos'].copy() + np.array([6.5e6, 0, 0]) 
    
    dv_vec = np.array(dv_values) * 1000.0 
    start_vel = earth['vel'].copy() + dv_vec
    
    dot, = ax.plot([], [], [], '*', color='yellow', ms=12, label='Ship')
    trail, = ax.plot([], [], [], '-', color='white', linewidth=1, alpha=0.5)
    
    spacecraft_list.append({
        'pos': start_pos, 'vel': start_vel,
        'dot': dot, 'trail': trail,
        'history': deque(maxlen=500),
        'id': len(spacecraft_list) + 1
    })
    active_ship_idx = len(spacecraft_list) - 1
    print(f"Launched! dV: {dv_values} km/s")

add_btn([0.895, 0.70, 0.07, 0.05], 'LAUNCH', fire_spacecraft, color='#006600', hover='#008800')

# --- BOTTOM CENTER (Sliders) ---
ax_slider = plt.axes([0.3, 0.05, 0.4, 0.02], facecolor='#333333')
t_slider = Slider(ax_slider, 'Day', 0, 1000, valinit=0, valstep=1, color='#00FFFF')
t_slider.label.set_color('white'); t_slider.label.set_fontsize(8)

ax_speed = plt.axes([0.3, 0.02, 0.4, 0.02], facecolor='#333333')
s_slider = Slider(ax_speed, 'Warp', 1, 86400, valinit=dt, color='#00FFFF')
s_slider.label.set_color('white'); s_slider.label.set_fontsize(8)

### ==========BOTTOM RIGHT (Flight Controls)========== ###

stats_text = ax.text2D(0.85, 0.16, "NO SIGNAL", transform=ax.transAxes, 
                       color='yellow', fontsize=9, family='monospace', fontweight='bold')

def cycle_ship(event):
    global active_ship_idx
    if not spacecraft_list: return
    active_ship_idx = (active_ship_idx + 1) % len(spacecraft_list)

def maneuver(direction):
    if active_ship_idx == -1 or not spacecraft_list: return
    ship = spacecraft_list[active_ship_idx]
    
    if focus_index == -1:
        rel_vel = ship['vel']
    else:
        rel_vel = ship['vel'] - ready_planets[focus_index]['vel']
        
    speed = np.linalg.norm(rel_vel)
    if speed < 1: return 
    
    norm_vel = rel_vel / speed
    burn_amount = 200.0 * direction 
    ship['vel'] += norm_vel * burn_amount
    print(f"Burn: {burn_amount} m/s")

add_btn([0.895, 0.10, 0.07, 0.04], 'CYCLE SHIP', cycle_ship, color='#444488')
add_btn([0.895, 0.05, 0.035, 0.04], 'PRO', lambda e: maneuver(1.0), color='#444444')
add_btn([0.935, 0.05, 0.035, 0.04], 'RET', lambda e: maneuver(-1.0), color='#444444')

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
    if event.key == 'p' or event.key == 'P': is_paused = not is_paused
    if event.key == '=' or event.key == '+': current_zoom *= 0.8
    if event.key == '-' or event.key == '_': current_zoom *= 1.2
    if event.key == 'up': maneuver(1.0)    
    if event.key == 'down': maneuver(-1.0) 
        
fig.canvas.mpl_connect('key_press_event', on_press)

def update(frame):
    global total_second_elapsed, dt, current_zoom
    
    if active_ship_idx != -1 and spacecraft_list:
        ship = spacecraft_list[active_ship_idx]
        
        if focus_index == -1: 
            ref_pos, ref_vel = np.zeros(3), np.zeros(3)
            ref_name = "Sun"
        else:
            ref_pos = ready_planets[focus_index]['pos']
            ref_vel = ready_planets[focus_index]['vel']
            ref_name = ready_planets[focus_index]['name']
            
        dist = np.linalg.norm(ship['pos'] - ref_pos)
        speed = np.linalg.norm(ship['vel'] - ref_vel)
        
        # Telemetry: Using Earth Radius (6.371e6) to show approximate Altitude
        if ref_name == 'Earth':
            alt_km = (dist - 6.371e6) / 1000.0
            stats_text.set_text(f"SHIP {ship['id']} [Earth]\nAlt: {alt_km:,.0f} km\nVel: {speed/1000:.1f} km/s")
        else:
            stats_text.set_text(f"SHIP {ship['id']} [{ref_name}]\nDist: {dist/1e6:.1f} Mm\nVel: {speed/1000:.1f} km/s")
        stats_text.set_color('yellow')
        
        for i, s in enumerate(spacecraft_list):
            if i == active_ship_idx:
                s['dot'].set_color('yellow'); s['dot'].set_markersize(10)
            else:
                s['dot'].set_color('white'); s['dot'].set_markersize(5)
    else:
        stats_text.set_text("NO SHIP ACTIVE")
        stats_text.set_color('#555555')

    if is_paused: return 
    
    # --- SUB-STEPPING LOGIC ---
    remaining_dt = dt
    
    while remaining_dt > 0:
        step = min(remaining_dt, MAX_PHYSICS_DT)
        
        for i in range(len(ready_planets)):
            planet = ready_planets[i]
            if planet['name'] == 'Sun': continue 
            pos = planet['pos']; vel = planet['vel']
            r_mag = np.linalg.norm(pos)
            accel = -GM / (r_mag ** 3) * pos
            vel += accel * step
            pos += vel * step
            planet['pos'] = pos; planet['vel'] = vel

        for ship in spacecraft_list:
            pos = ship['pos']; vel = ship['vel']
            r_sun = -pos; d_sun = np.linalg.norm(r_sun)
            accel_total = (G * M_sun / d_sun**3) * r_sun
            
            for planet in ready_planets:
                if planet['name'] == 'Sun': continue
                r_vec = planet['pos'] - pos
                dist = np.linalg.norm(r_vec)
                if dist > 1e6: 
                    accel_total += (G * planet['mass'] / dist**3) * r_vec
            
            vel += accel_total * step
            pos += vel * step
            ship['pos'] = pos; ship['vel'] = vel
            
        remaining_dt -= step
        
    total_second_elapsed += dt
    time_text.set_text(format_time(total_second_elapsed))

    if focus_index == -1: offset = np.array([0.0, 0.0, 0.0])
    else: offset = ready_planets[focus_index]['pos'].copy()
    
    sun_disp = -offset; sun_dot.set_data([sun_disp[0]], [sun_disp[1]]); sun_dot.set_3d_properties([sun_disp[2]])
    
    l_y = 5.5e11 * current_zoom
    l_x = l_y * 2.2; l_z = 2e11 * current_zoom
    ax.set_xlim(-l_x, l_x); ax.set_ylim(-l_y, l_y); ax.set_zlim(-l_z, l_z)

    for i in range(len(ready_planets)):
        draw_pos = ready_planets[i]['pos'] - offset 
        visuals[i]['x_history'].append(draw_pos[0]); visuals[i]['y_history'].append(draw_pos[1]); visuals[i]['z_history'].append(draw_pos[2])
        visuals[i]['dot'].set_data([draw_pos[0]], [draw_pos[1]]); visuals[i]['dot'].set_3d_properties([draw_pos[2]])
        visuals[i]['trail'].set_data(list(visuals[i]['x_history']), list(visuals[i]['y_history'])); visuals[i]['trail'].set_3d_properties(list(visuals[i]['z_history']))

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
except: pass

ani = FuncAnimation(fig, update, frames=None, interval=20, blit=False, cache_frame_data=False)
plt.show()