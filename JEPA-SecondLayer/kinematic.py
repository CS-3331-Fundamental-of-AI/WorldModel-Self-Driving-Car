import json, numpy as np
import os
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing

# -----------------------------
# Utility: load metadata_i.json
# -----------------------------
def load_meta(meta_path, idx):
    fname = os.path.join(meta_path, f"metadata_{idx:04d}.json")
    with open(fname) as f:
        return json.load(f)

# -----------------------------
# 2. Bicycle propagation
# -----------------------------
def bicycle_step(x, y, psi, v, delta, a_long, dt, L=2.843):
    x_next   = x + v * np.cos(psi) * dt
    y_next   = y + v * np.sin(psi) * dt
    psi_next = psi + (v / L) * np.tan(delta) * dt
    v_next   = v + a_long * dt
    return x_next, y_next, psi_next, v_next


# -----------------------------
# 3. Compute 4-step rollout and errors
# -----------------------------
def compute_4_step_error(meta_path, start_idx):
    # Load frames 7 → 11
    metas = [load_meta(meta_path, start_idx+i) for i in range(5)]
    
    # Start from metadata_0007
    meta_t = metas[0]
    
    # Extract params from trajectory window at t
    est = estimate_from_traj_window(meta_t["trajectory_window"], meta_t)

    # Initial state
    x = est["prediction_t+1"]["x"]
    y = est["prediction_t+1"]["y"]
    psi = est["prediction_t+1"]["yaw"]
    v = est["prediction_t+1"]["v"]
    delta = est["steering_angle"]
    a_long = est["acceleration"]
    dt = 0.1 #est["dt"]

    # Rollout predictions for t+1 .. t+4
    preds = []
    for k in range(4):
        x, y, psi, v = bicycle_step(x, y, psi, v, delta, a_long, dt)
        preds.append({"x":x, "y":y, "psi":psi, "v":v})

    # Compare to real frames metadata_0008 .. metadata_0011
    errors = []
    for i in range(1,5):
        gt = metas[i]["ego_state"]
        px, py, ppsi, pv = preds[i-1]["x"], preds[i-1]["y"], preds[i-1]["psi"], preds[i-1]["v"]

        gt_x = gt["position"]["x"]
        gt_y = gt["position"]["y"]
        gt_psi = gt["rotation"]["yaw"]
        gt_v = gt["speed_ms"]

        errors.append({
            "step": i,
            "error_x": px - gt_x,
            "error_y": py - gt_y,
            "error_pos_dist": np.hypot(px - gt_x, py - gt_y),
            "error_yaw": np.arctan2(np.sin(ppsi-gt_psi), np.cos(ppsi-gt_psi)),
            "error_v": pv - gt_v
        })

    return preds, errors

def estimate_from_traj_window(traj_window, current_state, L=2.843):
    frames = traj_window["positions"]
    N = traj_window["num_frames"]
    curr = frames[0]
    
    x = curr["x"]
    y = curr["y"]
    psi = curr["yaw"]
    vx = curr["vx"]
    vy = curr["vy"]
    v = np.hypot(vx, vy)
    a_long = current_state['ego_state']['acceleration_ms2']

    if N == 1:
        # --- start-of-simulation prior ---
        yaw_rate = 0.0        # no evidence of turning yet
        delta    = 0.0        # straight wheels
        # a_long   = 0.0        # or take from ego_state["acceleration_ms2"]
        dt = 0.1  # or your sim dt
    else:
        # normal multi-frame case (like we wrote before)
        prev = frames[1]
        dt_us = curr["timestamp"] - prev["timestamp"]
        dt = abs(dt_us) / 1e6
        dt = max(dt, 1e-6)

        psi_prev = prev["yaw"]
        dpsi = np.arctan2(np.sin(psi - psi_prev),
                          np.cos(psi - psi_prev))
        yaw_rate = dpsi / dt

        if v > 0.2:
            delta = np.arctan((yaw_rate * L) / v)
        else:
            delta = 0.0

        v_prev = np.hypot(prev["vx"], prev["vy"])
        a_long = (v - v_prev) / dt

    # simple 1-step prediction
    x_next = x + v * np.cos(psi) * dt
    y_next = y + v * np.sin(psi) * dt
    psi_next = psi + yaw_rate * dt
    v_next = v + a_long * dt

    return {
        "yaw_rate": yaw_rate,
        "steering_angle": delta,
        "acceleration": a_long,
        "current_speed": v,
        "prediction_t+1": {
            "x": x_next,
            "y": y_next,
            "yaw": psi_next,
            "v": v_next,
        },
    }

def ego_state_to_sobal_state(ego_state):
    yaw = ego_state["rotation"]["yaw"]
    
    return {
        "x":  ego_state["position"]["x"],
        "y":  ego_state["position"]["y"],
        "ux": np.cos(yaw),
        "uy": np.sin(yaw),
        "s":  ego_state["speed_ms"],
    }

def extract_sobal_action(meta_data, dt=0.1):
    ego = meta_data["ego_state"]
    traj = meta_data["trajectory_window"]["positions"]

    # Current and previous yaw
    yaw_t = traj[-1]["yaw"]
    yaw_prev = traj[-2]["yaw"] if len(traj) >= 2 else yaw_t

    yaw_rate = (yaw_t - yaw_prev) / dt

    return {
        "accel": ego["acceleration_ms2"],  # longitudinal acceleration
        "rot":   yaw_rate                  # turn command
    }

def sobal_step(state, action, dt):
    x, y, ux, uy, s = state["x"], state["y"], state["ux"], state["uy"], state["s"]
    a0, a1 = action["accel"], action["rot"]

    # Update position
    x_next = x + s * ux * dt
    y_next = y + s * uy * dt

    # Update speed
    s_next = s + a0 * dt

    # Update direction vector
    ux_tmp = ux + a1 * dt * uy
    uy_tmp = uy + a1 * dt * (-ux)

    # Normalize
    norm = np.hypot(ux_tmp, uy_tmp)
    if norm < 1e-8:
        ux_next, uy_next = ux, uy
    else:
        ux_next, uy_next = ux_tmp / norm, uy_tmp / norm

    # calculate the yaw & dirrectional velociuty
    yaw_rate = a1 # approximate
    yaw = np.arctan2(uy, ux)
    # Velocity components
    vx = s_next * ux
    vy = s_next * uy

    return {"x": x_next, "y": y_next, "ux": ux_next, "uy": uy_next, "s": s_next, "yaw": yaw, "vx": vx, "vy": vy}

def sobal_predict_4_steps(meta_data, dt=0.1):
    # Convert current state
    state = ego_state_to_sobal_state(meta_data["ego_state"])

    # Extract action from yaw change
    action = extract_sobal_action(meta_data, dt=dt)

    predictions = []
    s = state.copy()

    for i in range(4):
        s = sobal_step(s, action, dt)
        predictions.append(s)

    return predictions

def compute_windows_safe(frames):
    """
    Safe delta computation across a backward window of frames.

    Handles:
        - zero or repeated timestamps
        - large dt jumps (caps them)
        - padding frames
        - dx/dy spikes
        - yaw wrap
        - stable dv computation
    """
    deltas = []
    N = len(frames)

    for i in range(N - 1):
        f0 = frames[i]     # later frame (t)
        f1 = frames[i+1]   # earlier frame (t-1)

        # -------- Basic sanity check --------
        valid = (
            f0 is not None and f1 is not None and
            "timestamp" in f0 and "timestamp" in f1
        )

        if not valid:
            deltas.append([0, 0, 0, 0, 0, 0])
            continue

        # -----------------------------------------------------
        # 1. Timestamp handling (safe)
        # -----------------------------------------------------
        dt = abs(f0["timestamp"] - f1["timestamp"]) / 1e6  # convert μs → s

        if dt < 1e-5:
            dt = 1e-3       # treat as 1ms → avoids division explosion

        if dt > 0.50:
            dt = 0.50       # cap dt at 0.5s (NuScenes max spacing is ≈0.1s)

        # -----------------------------------------------------
        # 2. World-frame displacement
        # -----------------------------------------------------
        dx = f0["x"] - f1["x"]
        dy = f0["y"] - f1["y"]

        # Cap absurd spatial jumps (caused by scene boundary crashes)
        if abs(dx) > 50 or abs(dy) > 50:
            dx = 0
            dy = 0

        # -----------------------------------------------------
        # 3. Speed from displacement (stable)
        # -----------------------------------------------------
        v0 = np.hypot(dx, dy) / dt

        # Compute next-speed for dv
        if i < N - 2:
            f2 = frames[i+2]
            if f2 is not None:
                dt2 = abs(f1["timestamp"] - f2["timestamp"]) / 1e6
                if dt2 < 1e-5:
                    dt2 = 1e-3
                if dt2 > 0.50:
                    dt2 = 0.50

                dx2 = f1["x"] - f2["x"]
                dy2 = f1["y"] - f2["y"]

                if abs(dx2) > 50 or abs(dy2) > 50:
                    dx2 = dy2 = 0

                v1 = np.hypot(dx2, dy2) / dt2
            else:
                v1 = v0
        else:
            v1 = v0

        dv = v0 - v1

        # -----------------------------------------------------
        # 4. Safe yaw difference (wrap-aware)
        # -----------------------------------------------------
        yaw0 = f0["yaw"]
        yaw1 = f1["yaw"]
        dyaw = np.arctan2(np.sin(yaw0 - yaw1), np.cos(yaw0 - yaw1))

        # Remove angle noise: Dyaw > 0.3 rad in 0.1s is unrealistic
        if abs(dyaw) > 0.8:
            dyaw = 0.0

        # -----------------------------------------------------
        # 5. Ego-frame motion (ds_forward, ds_side)
        # -----------------------------------------------------
        cy = np.cos(yaw1)
        sy = np.sin(yaw1)

        ds_forward =  cy * dx + sy * dy
        ds_side    = -sy * dx + cy * dy

        # -----------------------------------------------------
        # 6. Final delta vector
        # -----------------------------------------------------
        deltas.append([dx, dy, dv, dyaw, ds_forward, ds_side])

    return deltas

def load_gpickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)

def pad_traj(vecs, target_len, dim):
    """
    vecs: list or np.ndarray of shape (N, dim) or (dim,)
    target_len: desired number of delta vectors
    dim: feature dimension per delta
    """

    # ---- 1. Convert to numpy ----
    v = np.array(vecs)

    # ---- 2. Handle empty input ----
    if v.size == 0:
        # Return full-zero padded matrix
        return np.zeros((target_len, dim), dtype=float)

    # ---- 3. Ensure 2D shape ----
    if v.ndim == 1:
        # [dim] → [1, dim]
        v = v.reshape(1, dim)

    # ---- 4. Compute padding needed ----
    pad_count = target_len - len(v)

    # ---- 5. Case: too long (truncate) ----
    if pad_count < 0:
        return v[:target_len]

    # ---- 6. Case: need padding ----
    pad = np.zeros((pad_count, dim), dtype=float)

    return np.concatenate([v, pad], axis=0)

def make_zero_frame(example_frame):
    # example_frame is a dict {"x":..., "y":..., "yaw":..., "timestamp":...}
    zero = {k: 0.0 for k in example_frame.keys()}
    # But keep timestamp meaningful (0)
    zero["timestamp"] = 0
    return zero

def load_json_worker(path):
    with open(path, "r") as f:
        js = json.load(f)
    return js

def load_all_metadata_parallel(metadata_dir):
    metadata_dir = Path(metadata_dir)
    file_list = sorted(metadata_dir.glob("metadata_*.json"))

    if len(file_list) == 0:
        raise RuntimeError("No metadata json found.")

    num_workers = min( max(1, multiprocessing.cpu_count()-1), 16 )

    print(f"Loading {len(file_list)} metadata files using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        json_list = list(executor.map(load_json_worker, file_list))

    print(f"Loaded {len(json_list)} metadata files.")
    return json_list

def group_by_scene(json_list):
    scenes = {}  # scene_token -> {sample_token: js}
    for js in json_list:
        scene = js["sample_info"]["scene_token"]
        sample = js["sample_info"]["sample_token"]
        scenes.setdefault(scene, {})[sample] = js
    return scenes

def reconstruct_scene_worker(args):
    scene_token, token_map = args

    # find start token
    start_tok = None
    for tok, js in token_map.items():
        if js["navigation"]["prev_sample_token"] == "":
            start_tok = tok
            break

    if start_tok is None:
        raise RuntimeError(f"Scene {scene_token}: no starting frame.")

    # follow next_sample_token chain
    ordered = []
    tok = start_tok
    while tok != "": 
        js = token_map[tok]
        ordered.append(js)
        tok = js["navigation"]["next_sample_token"]

    return (scene_token, ordered)

def build_scene_mapping_parallel(metadata_dir):
    # 1. Load all files in parallel
    json_list = load_all_metadata_parallel(metadata_dir)

    # 2. Group by scene
    scenes = group_by_scene(json_list)

    # 3. Reconstruct each scene in parallel
    num_workers = min( max(1, multiprocessing.cpu_count()-1), 16 )

    print(f"Reconstructing {len(scenes)} scenes with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = executor.map(
            reconstruct_scene_worker,
            [(scene_token, token_map) for scene_token, token_map in scenes.items()]
        )

    scene_mapping = {scene_token: ordered for scene_token, ordered in results}

    print(f"Finished processing {len(scene_mapping)} scenes.")
    return scene_mapping

def compute_windows_safe(frames):
    """
    Safe delta computation across a backward window of frames.

    Handles:
        - zero or repeated timestamps
        - large dt jumps (caps them)
        - padding frames
        - dx/dy spikes
        - yaw wrap
        - stable dv computation
    """
    deltas = []
    N = len(frames)

    for i in range(N - 1):
        f0 = frames[i]     # later frame (t)
        f1 = frames[i+1]   # earlier frame (t-1)

        # -------- Basic sanity check --------
        valid = (
            f0 is not None and f1 is not None and
            "timestamp" in f0 and "timestamp" in f1
        )

        if not valid:
            deltas.append([0, 0, 0, 0, 0, 0])
            continue

        # -----------------------------------------------------
        # 1. Timestamp handling (safe)
        # -----------------------------------------------------
        dt = abs(f0["timestamp"] - f1["timestamp"]) / 1e6  # convert μs → s

        if dt < 1e-5:
            dt = 1e-3       # treat as 1ms → avoids division explosion

        if dt > 0.50:
            dt = 0.50       # cap dt at 0.5s (NuScenes max spacing is ≈0.1s)

        # -----------------------------------------------------
        # 2. World-frame displacement
        # -----------------------------------------------------
        dx = f0["x"] - f1["x"]
        dy = f0["y"] - f1["y"]

        # Cap absurd spatial jumps (caused by scene boundary crashes)
        if abs(dx) > 50 or abs(dy) > 50:
            dx = 0
            dy = 0

        # -----------------------------------------------------
        # 3. Speed from displacement (stable)
        # -----------------------------------------------------
        v0 = np.hypot(dx, dy) / dt

        # Compute next-speed for dv
        if i < N - 2:
            f2 = frames[i+2]
            if f2 is not None:
                dt2 = abs(f1["timestamp"] - f2["timestamp"]) / 1e6
                if dt2 < 1e-5:
                    dt2 = 1e-3
                if dt2 > 0.50:
                    dt2 = 0.50

                dx2 = f1["x"] - f2["x"]
                dy2 = f1["y"] - f2["y"]

                if abs(dx2) > 50 or abs(dy2) > 50:
                    dx2 = dy2 = 0

                v1 = np.hypot(dx2, dy2) / dt2
            else:
                v1 = v0
        else:
            v1 = v0

        dv = v0 - v1

        # -----------------------------------------------------
        # 4. Safe yaw difference (wrap-aware)
        # -----------------------------------------------------
        yaw0 = f0["yaw"]
        yaw1 = f1["yaw"]
        dyaw = np.arctan2(np.sin(yaw0 - yaw1), np.cos(yaw0 - yaw1))

        # Remove angle noise: Dyaw > 0.3 rad in 0.1s is unrealistic
        if abs(dyaw) > 0.8:
            dyaw = 0.0

        # -----------------------------------------------------
        # 5. Ego-frame motion (ds_forward, ds_side)
        # -----------------------------------------------------
        cy = np.cos(yaw1)
        sy = np.sin(yaw1)

        ds_forward =  cy * dx + sy * dy
        ds_side    = -sy * dx + cy * dy

        # -----------------------------------------------------
        # 6. Final delta vector
        # -----------------------------------------------------
        deltas.append([dx, dy, dv, dyaw, ds_forward, ds_side])

    return deltas