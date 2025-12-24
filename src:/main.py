import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from collections import deque
import copy
from datetime import datetime, timedelta

### ==========Constants========== ###
AU_TO_METERS = 149597870700
DAY_TO_SECONDS = 86400
G = 6.67430e-11
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
            physics_bodies.append({"name": body['name'], "pos": pos_m, "vel": vel_ms, "mass": mass_kg})
        return physics_bodies, start_time_obj
    except:
        return [{'name': 'Sun', 'pos': np.zeros(3), 'vel': np.zeros(3), 'mass': M_sun},
                {'name': 'Earth', 'pos': np.array([AU_TO_METERS, 0, 0]), 'vel': np.array([0, 29780, 0]), 'mass': 5.97e24},
                {'name': 'Mars', 'pos': np.array([1.52*AU_TO_METERS, 0, 1e9]), 'vel': np.array([0, 24000, 100]), 'mass': 6.39e23}], datetime.now()

ready_planets, start_time = load_physics_data()
initial_state = copy.deepcopy(ready_planets) 

### ==========Matplotlib Setup========== ###
fig = plt.figure(figsize=(16, 9), facecolor='black')
ax = fig.add_axes([0.15, 0.05, 0.70, 0.90], projection='3d')
ax.set_facecolor('black') 
ax.set_axis_off() 
ax.set_box_aspect((1, 1, 1)) 

# --- Grid Setup ---
grid_max = 20e11; grid_step = 2e11 
grid_floor_lines = []; grid_wall_lines = []; grid_labels = []

for x in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([x, x], [-grid_max, grid_max], [0], color='#333333', linewidth=0.6)
    grid_floor_lines.append(l)
for y in np.arange(-grid_max, grid_max + grid_step, grid_step):
    l, = ax.plot([-grid_max, grid_max], [y, y], [0], color='#333333', linewidth=0.6)
    grid_floor_lines.append(l)

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

ax_check = plt.axes([0.02, 0.15, 0.08, 0.1], facecolor='#333333')
check = CheckButtons(ax_check, ['Floor'], [True])
check.labels[0].set_color('white')
def toggle_grid(label):
    if label == 'Floor': [l.set_visible(not l.get_visible()) for l in grid_floor_lines]
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

# --- Launch Panel ---
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
    
    start_pos = earth['pos'].copy() + np.array([6.5e6, 0, 0]) 
    dv_vec = np.array(dv_values) * 1000.0 
    start_vel = earth['vel'].copy() + dv_vec
    
    dot, = ax.plot([], [], [], '*', color='yellow', ms=12, label='Ship')
    trail, = ax.plot([], [], [], '-', color='white', linewidth=1, alpha=0.5)
    
    spacecraft_list.append({
        'pos': start_pos, 'vel': start_vel,
        'dot': dot, 'trail': trail,
        'history': deque(maxlen=500),
        'id': len(spacecraft_list) + 1,
        'prev_mars_dist': 1e20
    })
    active_ship_idx = len(spacecraft_list) - 1
    print(f"Launched! dV: {dv_values} km/s")

add_btn([0.895, 0.70, 0.07, 0.05], 'LAUNCH', fire_spacecraft, color='#006600', hover='#008800')

# --- Sliders ---
ax_slider = plt.axes([0.3, 0.05, 0.4, 0.02], facecolor='#333333')
t_slider = Slider(ax_slider, 'Day', 0, 1000, valinit=0, valstep=1, color='#00FFFF')
t_slider.label.set_color('white'); t_slider.label.set_fontsize(8)

ax_speed = plt.axes([0.3, 0.02, 0.4, 0.02], facecolor='#333333')
s_slider = Slider(ax_speed, 'Warp', 1, 86400, valinit=dt, color='#00FFFF')
s_slider.label.set_color('white'); s_slider.label.set_fontsize(8)

### ==========HUD & LOGIC========== ###

stats_text = ax.text2D(0.85, 0.16, "NO SIGNAL", transform=ax.transAxes, 
                       color='yellow', fontsize=9, family='monospace', fontweight='bold')
window_text = ax.text2D(0.85, 0.22, "", transform=ax.transAxes, 
                        color='cyan', fontsize=9, family='monospace')
capture_text = ax.text2D(0.85, 0.28, "", transform=ax.transAxes, 
                        color='red', fontsize=9, family='monospace', fontweight='bold')
traj_text = ax.text2D(0.85, 0.34, "", transform=ax.transAxes, 
                      color='orange', fontsize=9, family='monospace')

def cycle_ship(event):
    global active_ship_idx
    if not spacecraft_list: return
    active_ship_idx = (active_ship_idx + 1) % len(spacecraft_list)

def maneuver(direction):
    if active_ship_idx == -1 or not spacecraft_list: return
    ship = spacecraft_list[active_ship_idx]
    
    if focus_index == -1: rel_vel = ship['vel']
    else: rel_vel = ship['vel'] - ready_planets[focus_index]['vel']
        
    speed = np.linalg.norm(rel_vel)
    if speed < 1: return 
    
    norm_vel = rel_vel / speed
    burn_amount = 200.0 * direction 
    ship['vel'] += norm_vel * burn_amount
    print(f"Burn: {burn_amount} m/s")

def auto_correct_path(event):
    """
    NASA-Grade Flight Computer:
    1. Scans for the optimal arrival time.
    2. Calculates a B-Plane Target (Safe Altitude + "Side" Offset).
    3. Uses a Shooting Method to refine the trajectory.
    """
    if active_ship_idx == -1 or not spacecraft_list: return
    ship = spacecraft_list[active_ship_idx]
    
    p_mars = next((p for p in ready_planets if p['name'] == 'Mars'), None)
    if not p_mars: return

    print("\n=== STARTING FLIGHT COMPUTER ===")
    
    # --- STEP 1: COARSE SCAN (Find Best Arrival Time) ---
    best_time = 0
    min_dist_guess = float('inf')
    
    # Scan from 60 days to 400 days in the future
    scan_times = range(60, 400, 20) 
    
    ship_pos_snap = ship['pos'].copy()
    ship_vel_snap = ship['vel'].copy()
    
    print("Step 1: Analyzing Launch Windows...")
    for days in scan_times:
        sim_dt = days * 86400
        sim_mars = p_mars['pos'].copy()
        sim_mars_vel = p_mars['vel'].copy()
        
        # Approximate Mars Position (Fast Euler)
        sub_step = sim_dt / 5
        for _ in range(5):
            r = np.linalg.norm(sim_mars)
            acc = -GM / (r**3) * sim_mars
            sim_mars_vel += acc * sub_step
            sim_mars += sim_mars_vel * sub_step
        
        # Approximate Ship Position (Fast Euler)
        sim_ship = ship_pos_snap.copy()
        sim_ship_vel = ship_vel_snap.copy()
        for _ in range(5):
            r = np.linalg.norm(sim_ship)
            acc = -GM / (r**3) * sim_ship
            sim_ship_vel += acc * sub_step
            sim_ship += sim_ship_vel * sub_step
            
        dist = np.linalg.norm(sim_ship - sim_mars)
        if dist < min_dist_guess:
            min_dist_guess = dist
            best_time = sim_dt

    print(f"Optimal Flight Time Est: {best_time/86400:.0f} days")
    
    # --- STEP 2: CALCULATE SAFETY OFFSET (B-Plane Logic) ---
    target_time = best_time
    correction_vel = np.zeros(3) 
    
    # Predict where Mars and Ship will roughly be
    mars_future = p_mars['pos'] + p_mars['vel'] * target_time
    ship_future = ship['pos'] + ship['vel'] * target_time
    
    # 1. Find Incoming Velocity Vector (Direction to Mars)
    v_incoming = ship_future - mars_future
    v_incoming_norm = np.linalg.norm(v_incoming)
    if v_incoming_norm == 0: v_incoming = np.array([1.0, 0, 0])
    else: v_incoming = v_incoming / v_incoming_norm
    
    # 2. Cross Product for "Side" Offset
    # This ensures we aim 90 degrees to the right of our approach
    up_vector = np.array([0, 0, 1])
    offset_dir = np.cross(v_incoming, up_vector)
    
    # Safety: If coming in vertically, switch reference vector
    if np.linalg.norm(offset_dir) < 0.1:
        offset_dir = np.cross(v_incoming, np.array([1, 0, 0]))
        
    offset_dir = offset_dir / np.linalg.norm(offset_dir)
    
    # 3. Apply Safety Radius (6,000 km)
    # Radius of Mars is ~3,400 km. This gives ~2,600 km altitude.
    final_offset_vec = offset_dir * 6.0e6
    
    # --- STEP 3: ITERATIVE SHOOTING METHOD ---
    print("Step 2: Refining Trajectory...")
    
    for iteration in range(6): 
        # Clone Universe for High-Fidelity Simulation
        sim_planets = copy.deepcopy(ready_planets)
        sim_ship_pos = ship['pos'].copy()
        sim_ship_vel = ship['vel'].copy() + correction_vel
        
        # Simulation Parameters
        step_size = 3600 * 6 # 6-hour steps for speed/accuracy balance
        steps = int(target_time / step_size)
        
        # Run Physics Loop
        for _ in range(steps):
            # Move Planets
            for p in sim_planets:
                if p['name'] == 'Sun': continue
                r = np.linalg.norm(p['pos'])
                acc = -GM / (r**3) * p['pos']
                p['vel'] += acc * step_size
                p['pos'] += p['vel'] * step_size
            
            # Move Ship (Sun Gravity)
            r_sun = -sim_ship_pos
            acc_ship = (GM / np.linalg.norm(r_sun)**3) * r_sun
            sim_ship_vel += acc_ship * step_size
            sim_ship_pos += sim_ship_vel * step_size
            
        # Check Error
        sim_mars = next(p for p in sim_planets if p['name'] == 'Mars')
        
        # TARGET = Future Mars Position + Safety Offset
        target_point = sim_mars['pos'] + final_offset_vec
        error_vec = target_point - sim_ship_pos
        miss_dist = np.linalg.norm(error_vec)
        
        print(f"Iter {iteration+1}: Deviation {miss_dist/1e6:.1f} Mm")
        
        # Convergence Check (Hit within 500km of target point)
        if miss_dist < 0.5e6: 
            print("TARGET LOCK ACQUIRED.")
            break
            
        # Apply Linear Correction (dV = Error / Time)
        dv_adjustment = error_vec / target_time
        correction_vel += dv_adjustment
    
    # --- STEP 4: EXECUTE BURN ---
    ship['vel'] += correction_vel
    print(f"== BURN COMPLETE: {np.linalg.norm(correction_vel)/1000:.3f} km/s ==")

    # === NEW: AUTOMATIC ORBIT INSERTION ===
def auto_orbit(event):
    if active_ship_idx == -1 or not spacecraft_list: return
    ship = spacecraft_list[active_ship_idx]
    p_mars = next((p for p in ready_planets if p['name'] == 'Mars'), None)
    
    # Safety Check: Must be close to Mars
    dist = np.linalg.norm(ship['pos'] - p_mars['pos'])
    if dist > 100e6:
        print("ERROR: Too far from Mars! Wait until < 100 Mm.")
        return

    print("INITIATING ORBITAL INSERTION...")
    
    # 1. Calculate ideal Circular Velocity at current distance
    # V_circ = sqrt(GM / r)
    needed_speed_mag = np.sqrt(G * p_mars['mass'] / dist)
    
    # 2. Get current velocity Direction (Tangent to orbit)
    # Relative Velocity
    rel_vel = ship['vel'] - p_mars['vel']
    
    # We want to keep the Direction but clamp the Speed
    # (Simple normalization)
    current_speed_mag = np.linalg.norm(rel_vel)
    vel_dir = rel_vel / current_speed_mag
    
    # 3. Apply the perfect speed
    new_rel_vel = vel_dir * needed_speed_mag
    
    # 4. Set Ship Velocity
    ship['vel'] = p_mars['vel'] + new_rel_vel
    
    print(f"ORBIT ESTABLISHED! Altitude: {(dist - 3.4e6)/1000:.0f} km")



    # Target refinement (Offset Calculation)
    target_time = best_time
    correction_vel = np.zeros(3) 
    
    # Calculate Perpendicular Offset Vector
    # We want to aim 5000km to the 'side' of Mars, not the center
    # Cross product of Mars Vel and Up(Z) gives a safe planar side vector
    mars_v_norm = p_mars['vel'] / np.linalg.norm(p_mars['vel'])
    offset_dir = np.cross(mars_v_norm, np.array([0,0,1]))
    if np.linalg.norm(offset_dir) < 0.1: offset_dir = np.array([0,1,0]) # Fallback
    offset_vec = offset_dir * 6.0e6 # 6000 km Safety Radius
    
    for iteration in range(6): 
        sim_planets = copy.deepcopy(ready_planets)
        sim_ship_pos = ship['pos'].copy()
        sim_ship_vel = ship['vel'].copy() + correction_vel
        
        step_size = 3600 * 6 
        steps = int(target_time / step_size)
        
        for _ in range(steps):
            for p in sim_planets:
                if p['name'] == 'Sun': continue
                r = np.linalg.norm(p['pos'])
                acc = -GM / (r**3) * p['pos']
                p['vel'] += acc * step_size
                p['pos'] += p['vel'] * step_size
            
            r_sun = -sim_ship_pos
            acc_ship = (GM / np.linalg.norm(r_sun)**3) * r_sun
            sim_ship_vel += acc_ship * step_size
            sim_ship_pos += sim_ship_vel * step_size
            
        sim_mars = next(p for p in sim_planets if p['name'] == 'Mars')
        
        # AIM FOR MARS + OFFSET
        target_point = sim_mars['pos'] + offset_vec
        error_vec = target_point - sim_ship_pos
        miss_dist = np.linalg.norm(error_vec)
        
        print(f"Iter {iteration+1}: Miss {miss_dist/1e6:.1f} Mm")
        
        if miss_dist < 0.5e6: 
            print("TARGET LOCK ACQUIRED.")
            break
            
        dv_adjustment = error_vec / target_time
        correction_vel += dv_adjustment
    
    ship['vel'] += correction_vel
    print(f"== BURN COMPLETE: {np.linalg.norm(correction_vel)/1000:.3f} km/s ==")

# Buttons
add_btn([0.895, 0.10, 0.07, 0.04], 'CYCLE SHIP', cycle_ship, color='#444488')
add_btn([0.895, 0.05, 0.035, 0.04], 'PRO', lambda e: maneuver(1.0), color='#444444')
add_btn([0.935, 0.05, 0.035, 0.04], 'RET', lambda e: maneuver(-1.0), color='#444444')
add_btn([0.820, 0.05, 0.07, 0.04], 'FIX PATH', auto_correct_path, color='#AA4400', hover='#CC5500')
add_btn([0.820, 0.10, 0.07, 0.04], 'AUTO ORBIT', auto_orbit, color='#880000', hover='#AA0000')

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
    
    p_earth = next((p for p in ready_planets if p['name'] == 'Earth'), None)
    p_mars = next((p for p in ready_planets if p['name'] == 'Mars'), None)
    
    # 1. WINDOW CALCULATION
    if p_earth and p_mars:
        pos_e = p_earth['pos']; pos_m = p_mars['pos']
        ang_e = np.degrees(np.arctan2(pos_e[1], pos_e[0]))
        ang_m = np.degrees(np.arctan2(pos_m[1], pos_m[0]))
        phase_diff = (ang_m - ang_e) % 360
        diff = (phase_diff - 44.0) % 360
        days_est = diff / 0.461 
        
        r1 = np.linalg.norm(pos_e); r2 = np.linalg.norm(pos_m)
        v_inf = (np.sqrt(GM / r1) * np.sqrt((2 * r2) / (r1 + r2))) - np.sqrt(GM / r1)
        v_inj = np.sqrt(v_inf**2 + (2 * (G * 5.972e24) / 6.5e6))
        dv_launch = v_inj / 1000.0
        
        if 42 < phase_diff < 46:
            window_text.set_text(f"MARS WINDOW: LAUNCH NOW!\nRec. dV: {dv_launch:.1f} km/s")
            window_text.set_color("#00FF00")
        else:
            window_text.set_text(f"MARS WINDOW: WAIT\nEst. Days: {days_est:.0f}")
            window_text.set_color("cyan")

    # 2. TELEMETRY & PREDICTOR
    capture_text.set_text("") 
    traj_text.set_text("")
    
    if active_ship_idx != -1 and spacecraft_list:
        ship = spacecraft_list[active_ship_idx]
        if focus_index == -1: ref_pos, ref_vel, ref_name = np.zeros(3), np.zeros(3), "Sun"
        else:
            ref_pos = ready_planets[focus_index]['pos']
            ref_vel = ready_planets[focus_index]['vel']
            ref_name = ready_planets[focus_index]['name']
            
        dist = np.linalg.norm(ship['pos'] - ref_pos)
        speed = np.linalg.norm(ship['vel'] - ref_vel)
        
        # --- NEW: FUTURE PERIAPSIS PREDICTOR ---
        # Run a "Fast" check every 30 frames to guess where we will end up
        if p_mars and frame % 30 == 0:
            # Quick lookahead
            pred_mars_pos = p_mars['pos'].copy()
            pred_ship_pos = ship['pos'].copy()
            pred_ship_vel = ship['vel'].copy()
            
            min_future_dist = float('inf')
            
            # Simulate 200 days ahead in huge chunks
            # Only do this if ship is within 500Mm of Mars (Optimization)
            if np.linalg.norm(ship['pos'] - p_mars['pos']) < 500e6:
                steps_pred = 100
                dt_pred = (200 * 86400) / steps_pred
                for _ in range(steps_pred):
                    # Approx Ship Move (Kepler)
                    r = np.linalg.norm(pred_ship_pos)
                    a = -GM/(r**3) * pred_ship_pos
                    pred_ship_vel += a * dt_pred
                    pred_ship_pos += pred_ship_vel * dt_pred
                    
                    # Approx Mars Move (Linear velocity approx for UI speed)
                    pred_mars_pos += p_mars['vel'] * dt_pred
                    
                    d = np.linalg.norm(pred_ship_pos - pred_mars_pos)
                    if d < min_future_dist: min_future_dist = d
                
                ship['est_periapsis'] = min_future_dist

        # Display Prediction
        est_peri = ship.get('est_periapsis', 0)
        if est_peri > 0:
            p_alt = (est_peri - 3.4e6) / 1000.0
            if p_alt < 0: traj_str = "IMPACT DETECTED"
            else: traj_str = f"Est. Periapsis: {p_alt:,.0f} km"
            traj_text.set_text(traj_str)
            traj_text.set_color("orange" if p_alt > 2000 else "red")

        # --- CAPTURE LOGIC ---
        if p_mars:
            dist_mars = np.linalg.norm(ship['pos'] - p_mars['pos'])
            if dist_mars < 300e6:
                prev_dist = ship.get('prev_mars_dist', 1e20)
                ship['prev_mars_dist'] = dist_mars
                
                needed_speed = np.sqrt(G * p_mars['mass'] / dist_mars)
                vel_rel_mars = ship['vel'] - p_mars['vel']
                curr_speed_mars = np.linalg.norm(vel_rel_mars)
                capture_dv = (curr_speed_mars - needed_speed) / 1000.0
                
                if dist_mars > prev_dist: 
                    status = "PERIAPSIS REACHED!\nBURN RETRO NOW!"
                    col = "red"
                else: 
                    status = "APPROACHING (COAST)"
                    col = "orange"
                    
                capture_text.set_text(f"== MARS APPROACH ==\n{status}\nDist: {dist_mars/1e6:.1f} Mm\nCapture dV: {capture_dv:.1f} km/s")
                capture_text.set_color(col)

        if ref_name == 'Earth':
            stats_text.set_text(f"SHIP {ship['id']} [Earth]\nAlt: {(dist - 6.371e6)/1000:.0f} km\nVel: {speed/1000:.1f} km/s")
        else:
            stats_text.set_text(f"SHIP {ship['id']} [{ref_name}]\nDist: {dist/1e6:.1f} Mm\nVel: {speed/1000:.1f} km/s")
        
        for i, s in enumerate(spacecraft_list):
            s['dot'].set_color('yellow' if i == active_ship_idx else 'white')
            s['dot'].set_markersize(10 if i == active_ship_idx else 5)
    else:
        stats_text.set_text("NO SHIP ACTIVE")

    if is_paused: return 

    # 3. PHYSICS LOOP
    remaining_dt = dt
    current_precision = 100 
    
    for ship in spacecraft_list:
        for p in ready_planets:
            if p['name'] == 'Sun': continue
            if np.linalg.norm(ship['pos'] - p['pos']) < 50e6:
                current_precision = 1 
                break
    
    while remaining_dt > 0:
        step = min(remaining_dt, current_precision)
        
        for p in ready_planets:
            if p['name'] == 'Sun': continue 
            r_mag = np.linalg.norm(p['pos'])
            accel = -GM / (r_mag ** 3) * p['pos']
            p['vel'] += accel * step
            p['pos'] += p['vel'] * step

        for ship in spacecraft_list:
            r_sun = -ship['pos']
            accel_total = (G * M_sun / np.linalg.norm(r_sun)**3) * r_sun
            for p in ready_planets:
                if p['name'] == 'Sun': continue
                r_vec = p['pos'] - ship['pos']
                d = np.linalg.norm(r_vec)
                if d < 1e9: accel_total += (G * p['mass'] / d**3) * r_vec
            
            ship['vel'] += accel_total * step
            ship['pos'] += ship['vel'] * step
        remaining_dt -= step
        
    total_second_elapsed += dt
    time_text.set_text(format_time(total_second_elapsed))

    # 4. DRAW
    if focus_index == -1: offset = np.array([0.0, 0.0, 0.0])
    else: offset = ready_planets[focus_index]['pos'].copy()
    
    sun_disp = -offset
    sun_dot.set_data([sun_disp[0]], [sun_disp[1]]); sun_dot.set_3d_properties([sun_disp[2]])
    
    for i, p in enumerate(ready_planets):
        draw_pos = p['pos'] - offset 
        visuals[i]['dot'].set_data([draw_pos[0]], [draw_pos[1]]); visuals[i]['dot'].set_3d_properties([draw_pos[2]])
        
        if frame % 5 == 0:
            visuals[i]['x_history'].append(draw_pos[0]); visuals[i]['y_history'].append(draw_pos[1]); visuals[i]['z_history'].append(draw_pos[2])
            visuals[i]['trail'].set_data(list(visuals[i]['x_history']), list(visuals[i]['y_history'])); visuals[i]['trail'].set_3d_properties(list(visuals[i]['z_history']))

    for ship in spacecraft_list:
        draw_pos = ship['pos'] - offset
        ship['dot'].set_data([draw_pos[0]], [draw_pos[1]]); ship['dot'].set_3d_properties([draw_pos[2]])
        if frame % 2 == 0:
            ship['history'].append(draw_pos); hist = np.array(ship['history'])
            if len(hist) > 0:
                ship['trail'].set_data(hist[:,0], hist[:,1]); ship['trail'].set_3d_properties(hist[:,2])
                
    max_range = 5.5e11 * current_zoom
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

def format_time(seconds):
    current_date = start_time + timedelta(seconds=seconds)
    return current_date.strftime('Mission Time: %Y-%m-%d %H:%M')

try:
    mng = plt.get_current_fig_manager()
    if hasattr(mng, 'resize') and hasattr(mng, 'window'): mng.resize(*mng.window.maxsize())
except: pass

ani = FuncAnimation(fig, update, frames=None, interval=20, blit=False, cache_frame_data=False)
plt.show()