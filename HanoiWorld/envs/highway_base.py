"""
Base Highway Environment wrapper for DreamerV3.
Contains the main HighwayEnv class with all core functionality.
"""
import gymnasium
import numpy as np
# Import highway_env to register the environments with gymnasium
import highway_env
from .highway_rewards import DEFAULT_REWARD_CONFIGS, get_reward_config


# Mapping from short names to full environment names
ENV_NAME_MAPPING = {
    "highway": "highway-v0",
    "intersection": "intersection-v1",
    "parking": "parking-v0",
    "merge": "merge-v0",
    "roundabout": "roundabout-v0",
    "racetrack": "racetrack-v0",
    "twowayhighway": "two-way-v0",
    "localhighway": "local-highway-v0",
    "localhighway_fast": "local-highway-fast-v0",
    "localhighway_hetero": "local-highway-hetero-v0",
    "localhighway_hetero_h": "local-highway-hetero-H-v0",
    "localhighway_hetero_vh": "local-highway-hetero-VH-v0",
    "localhighway_mo": "local-mo-highway-v0",
}


class HighwayEnv(gymnasium.Env):
    """
    Wrapper for Highway-Env autonomous driving environments.
    Supports highway-v0, intersection-v1, parking-v0, merge-v0, roundabout-v0, racetrack-v0.
    
    Highway-env documentation: https://highway-env.farama.org/
    
    Features:
    - Image or kinematics observations
    - Discrete or continuous actions
    - Configurable reward shaping
    - Lane-neutral behavior (no lane preference bias)
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(
        self,
        name,
        render_mode="rgb_array",
        action_repeat=1,
        size=(64, 64),
        obs_type="image",
        action_type="discrete",
        seed=None,
        use_reward_shaping=True,
        reward_config=None,
        vehicles_count=50,
        vehicles_density=1.5,
        offscreen_rendering=True,
    ):
        """
        Args:
            name: Environment name (e.g., 'highway', 'intersection', 'parking')
            action_repeat: Number of times to repeat each action
            size: Image observation size (width, height)
            obs_type: Type of observation - 'image', 'kinematics', or 'grayscale'
            action_type: Type of action space - 'discrete' or 'continuous'
            seed: Random seed
            use_reward_shaping: Whether to use advanced reward shaping
            reward_config: Custom reward configuration dict (overrides defaults)
            vehicles_count: Number of vehicles in the environment
            vehicles_density: Density of vehicles on road
        """
        super().__init__()
        self._name = name
        self._render_mode = render_mode
        self._action_repeat = action_repeat
        self._size = size
        self._obs_type = obs_type
        self._action_type = action_type
        self._use_reward_shaping = use_reward_shaping
        self._vehicles_count = vehicles_count
        self._vehicles_density = vehicles_density
        self._use_rgb_render = False
        self._offscreen_rendering = offscreen_rendering
        self.reward_range = [-np.inf, np.inf]
        self.off_road_count = 0
        self._was_on_road = True
        self._just_went_off_road = False

        self.prev_acc = 0.0
        self.prev_steer = 0.0
        self.jerk_sum = 0.0
        self.steer_rate_sum = 0.0
        self.lat_offsets = []
        self.comfort_steps = 0
        
        # Setup reward configuration
        self._reward_config = get_reward_config(name, reward_config)
        
        # Initialize tracking variables for reward shaping
        self._reset_tracking_variables()
        
        # Create and configure environment
        # disable_env_checker=True to suppress warnings about obs space mismatch
        # (we use minimal Kinematics config but return image observations from render())
        env_name = self._get_env_name(name)
        self._env = gymnasium.make(env_name, render_mode=self._render_mode, disable_env_checker=True)
        self._configure_env()
        
        if seed is not None:
            self._env.reset(seed=seed)
        
        self._done = True

    def _reset_tracking_variables(self):
        """Reset all tracking variables used for reward shaping."""
        self._last_lane_index = None
        self._last_position = None
        self._blocked_by_slow = False
        self._was_blocked = False
        self._vehicles_ahead_ids = set()
        self.off_road_count = 0
        self._was_on_road = True
        self._just_went_off_road = False

    def _get_env_name(self, name):
        """Map short task name to full environment name."""
        if "-v" in name:
            return name
        return ENV_NAME_MAPPING.get(name, f"{name}-v0")

    def _configure_env(self):
        """Configure the highway environment."""
        config = self._get_base_config()
        config.update(self._get_env_specific_config())
        config.update(self._get_observation_config())
        
        self._env.unwrapped.configure(config)
        self._env.reset()

    def _get_base_config(self):
        """Get base configuration shared by all environments."""
        return {
            "action": {
                "type": "DiscreteMetaAction" if self._action_type == "discrete" else "ContinuousAction",
            },
            "offscreen_rendering": self._offscreen_rendering,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            "duration": 40,
            "screen_width": 600,
            "screen_height": 300,
            "scaling": 5.5,
            # Override highway-env's default reward weights to be LANE NEUTRAL
            # We handle reward shaping ourselves in _compute_shaped_reward
            "right_lane_reward": 0.0,  # CRITICAL: disable built-in right lane bias!
            "lane_change_reward": 0.0,
        }

    def _get_env_specific_config(self):
        """Get environment-specific configuration."""
        env_name = self._name.lower()
        config = {}
        
        if env_name in ("highway", "highway-v0"):
            config["vehicles_count"] = self._vehicles_count
            config["vehicles_density"] = self._vehicles_density
            config["lanes_count"] = 4
            
        elif env_name in ("intersection", "intersection-v0", "intersection-v1"):
            config["initial_vehicle_count"] = min(self._vehicles_count, 20)
            config["spawn_probability"] = min(self._vehicles_density / 2.0, 0.8)
            
        elif env_name in ("merge", "merge-v0"):
            config["duration"] = 50
            
        elif env_name in ("roundabout", "roundabout-v0"):
            config["duration"] = 15
            
        elif env_name in ("racetrack", "racetrack-v0"):
            config["other_vehicles"] = min(self._vehicles_count, 10)
            config["controlled_vehicles"] = 1
            if self._action_type == "continuous":
                config["action"] = {
                    "type": "ContinuousAction",
                    "longitudinal": True,
                    "lateral": True
                }
        else:
            config["vehicles_count"] = self._vehicles_count
            config["vehicles_density"] = self._vehicles_density
            
        return config

    def _get_observation_config(self):
        """Get observation configuration."""
        config = {}
        
        if self._obs_type == "image":
            # For RGB image observations, we use env.render() directly
            # Use minimal "Kinematics" observation to avoid wasting computation
            # on GrayscaleObservation that we would ignore anyway
            config["observation"] = {
                "type": "Kinematics",
                "vehicles_count": 1,  # Minimal - we don't use this
                "features": ["presence"],
                "normalize": False,
            }
            self._use_rgb_render = True
            
        elif self._obs_type == "grayscale":
            # For grayscale, we could use highway-env's GrayscaleObservation
            # but for consistency, we also use render() and convert
            config["observation"] = {
                "type": "Kinematics",
                "vehicles_count": 1,
                "features": ["presence"],
                "normalize": False,
            }
            # Use render and convert to grayscale ourselves
            self._use_rgb_render = True
            
        elif self._obs_type == "kinematics":
            config["observation"] = {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
            }
            self._use_rgb_render = False
            
        return config

    # =========================================================================
    # Reward Shaping
    # =========================================================================
    
    def _compute_shaped_reward(self, base_reward, info, terminated, truncated):
        """Compute shaped reward with multiple components."""
        if not self._use_reward_shaping:
            return base_reward
        
        rc = self._reward_config
        reward = 0.0
        
        try:
            ego_vehicle = self._env.unwrapped.vehicle
            if ego_vehicle is None:
                return base_reward
        except (AttributeError, TypeError):
            return base_reward
        
        # 1. Collision penalty (highest priority)
        if terminated and info.get("crashed", getattr(ego_vehicle, 'crashed', False)):
            return rc.get("collision_reward", -2.0)
        
        # 2. Speed reward
        reward += self._compute_speed_reward(ego_vehicle, rc)
        
        # 3. Safe distance penalty
        reward += self._compute_distance_penalty(ego_vehicle, rc)
        
        # 4. Lane change rewards (smart lane changes, overtaking)
        reward += self._compute_lane_rewards(ego_vehicle, rc)
        
        # 5. On-road reward
        reward += self._compute_on_road_reward(ego_vehicle, rc)
        
        # 6. Progress reward
        reward += self._compute_progress_reward(ego_vehicle, rc)
        
        # 7. Heading alignment
        reward += self._compute_heading_reward(ego_vehicle, rc)
        
        # 8. Lane centering / lateral deviation
        res, _ = self._compute_lateral_deviation_reward(ego_vehicle, rc)
        reward += res
        
        # 9. Terminal rewards
        if terminated or truncated:
            if not info.get("crashed", False):
                reward += rc.get("success_reward", 0.3)
        
        # Blend with original reward
        blend_factor = rc.get("shaped_reward_weight", 0.85)
        final_reward = blend_factor * reward + (1 - blend_factor) * base_reward
        
        return float(np.clip(final_reward, -2.0, 2.0))

    def _compute_speed_reward(self, ego_vehicle, rc):
        """Compute speed-based reward."""
        try:
            speed = getattr(ego_vehicle, 'speed', 0.0)
            speed_range = rc.get("reward_speed_range", [22, 23])
            min_speed, max_speed = speed_range
            high_speed_reward = rc.get("high_speed_reward", 0.6)
            
            if speed < min_speed:
                return (speed / min_speed) * high_speed_reward
            elif speed <= max_speed:
                return high_speed_reward
            else:
                overspeed_ratio = (speed - max_speed) / max_speed
                return high_speed_reward * max(0, 1 - overspeed_ratio * 0.5)
        except (AttributeError, TypeError):
            return 0.0

    def _compute_distance_penalty(self, ego_vehicle, rc):
        """
        Compute penalty for being too close to vehicles that matter.
        
        Only penalizes:
        - Vehicles AHEAD of ego (not behind)
        - Vehicles in SAME or ADJACENT lanes (not 2+ lanes away)
        
        Uses separate longitudinal and lateral distance thresholds.
        """
        try:
            road = self._env.unwrapped.road
            if road is None or not hasattr(road, 'vehicles'):
                return 0.0
            
            ego_pos = np.array(ego_vehicle.position)
            ego_lane = getattr(ego_vehicle, 'lane_index', None)
            ego_speed = getattr(ego_vehicle, 'speed', 25.0)
            
            # Distance thresholds
            min_longitudinal_distance = rc.get("min_safe_distance", 15.0)
            min_lateral_distance = rc.get("min_lateral_distance", 2.0)  # Lane width ~4m, so 2m is close
            
            # Speed-dependent longitudinal threshold (faster = need more distance)
            speed_factor = max(0.5, ego_speed / 25.0)  # Scale by speed
            adjusted_long_dist = min_longitudinal_distance * speed_factor
            
            max_penalty = 0.0
            
            for vehicle in road.vehicles:
                if vehicle is ego_vehicle:
                    continue
                
                v_pos = np.array(vehicle.position)
                v_lane = getattr(vehicle, 'lane_index', None)
                
                # Compute longitudinal (x) and lateral (y) distances
                dx = v_pos[0] - ego_pos[0]  # Positive = vehicle is ahead
                dy = abs(v_pos[1] - ego_pos[1])  # Lateral distance
                
                # Only consider vehicles AHEAD of us (dx > 0)
                # Ignore vehicles behind us - they're their problem!
                if dx <= 0:
                    continue
                
                # Check if vehicle is in same or adjacent lane
                is_relevant_lane = True
                if ego_lane is not None and v_lane is not None:
                    lane_diff = abs(ego_lane[2] - v_lane[2])
                    # Only consider same lane (0) or adjacent lanes (1)
                    # Ignore vehicles 2+ lanes away
                    if lane_diff > 1:
                        is_relevant_lane = False
                
                if not is_relevant_lane:
                    continue
                
                # Compute penalty based on longitudinal distance (for vehicles ahead in relevant lanes)
                if dx < adjusted_long_dist:
                    # Longitudinal penalty - closer = much worse (quadratic)
                    long_ratio = 1 - (dx / adjusted_long_dist)
                    long_ratio = long_ratio ** 2  # sharper near 0
                    
                    # Lateral factor - same lane is worse than adjacent
                    if dy < min_lateral_distance:
                        # Very close laterally (same lane or cutting in)
                        lateral_factor = 1.0
                    elif dy < min_lateral_distance * 2:
                        # Adjacent lane - reduced penalty
                        lateral_factor = 0.5
                    else:
                        # Far enough laterally
                        lateral_factor = 0.2
                    
                    base_penalty = rc.get("safe_distance_penalty", 0.4)
                    
                    # Emergency zone: very close in same lane
                    if dx < min_longitudinal_distance * 0.5 and dy < min_lateral_distance:
                        base_penalty *= 3.0  # triple penalty near-collision
                    
                    penalty = long_ratio * lateral_factor * base_penalty
                    max_penalty = max(max_penalty, penalty)
            
            return -max_penalty
            
        except (AttributeError, TypeError):
            return 0.0

    def _compute_lane_rewards(self, ego_vehicle, rc):
        """Compute lane-related rewards (blocking penalty, smart lane change, overtake)."""
        reward = 0.0
        
        try:
            road = self._env.unwrapped.road
            ego_speed = getattr(ego_vehicle, 'speed', 0)
            ego_lane = getattr(ego_vehicle, 'lane_index', None)
            ego_pos = np.array(ego_vehicle.position) if hasattr(ego_vehicle, 'position') else None
            
            if road is None or ego_lane is None or ego_pos is None:
                return 0.0
            
            look_ahead = rc.get("look_ahead_distance", 50.0)
            slow_threshold = rc.get("slow_vehicle_threshold", 0.85)
            
            # Find vehicles ahead and nearby
            vehicle_ahead = None
            min_dist_ahead = float('inf')
            vehicles_nearby = []
            
            for vehicle in road.vehicles:
                if vehicle is ego_vehicle:
                    continue
                
                v_pos = np.array(vehicle.position)
                v_lane = getattr(vehicle, 'lane_index', None)
                dist_x = v_pos[0] - ego_pos[0]
                
                if -20 < dist_x < look_ahead:
                    vehicles_nearby.append((vehicle, dist_x, v_lane))
                
                # Check same lane
                if v_lane and v_lane[0] == ego_lane[0] and v_lane[1] == ego_lane[1] and v_lane[2] == ego_lane[2]:
                    if 0 < dist_x < look_ahead and dist_x < min_dist_ahead:
                        min_dist_ahead = dist_x
                        vehicle_ahead = vehicle
            
            # Check if blocked
            is_blocked = False
            if vehicle_ahead is not None:
                v_speed = getattr(vehicle_ahead, 'speed', 0)
                if v_speed < ego_speed * slow_threshold or min_dist_ahead < 25:
                    is_blocked = True
                    closeness_factor = max(0.5, 1.0 - min_dist_ahead / look_ahead)
                    reward -= rc.get("blocked_lane_penalty", 0.3) * closeness_factor
            
            # Smart lane change reward
            was_blocked = self._blocked_by_slow
            self._blocked_by_slow = is_blocked
            
            if was_blocked and self._last_lane_index is not None:
                if ego_lane[2] != self._last_lane_index[2]:
                    reward += rc.get("smart_lane_change_reward", 0.2)
            
            # Overtake reward
            current_ahead_ids = set()
            for v, dist_x, v_lane in vehicles_nearby:
                if dist_x > 0:
                    current_ahead_ids.add(id(v))
                elif dist_x < -5:
                    if id(v) in self._vehicles_ahead_ids:
                        reward += rc.get("overtake_reward", 0.2)
            self._vehicles_ahead_ids = current_ahead_ids
            
            # Update last lane
            self._last_lane_index = ego_lane
            
        except (AttributeError, TypeError):
            pass
        
        return reward

    def _compute_on_road_reward(self, ego_vehicle, rc):
        """Compute reward for staying on road."""
        try:
            self._update_off_road_count(ego_vehicle)
            on_road = getattr(ego_vehicle, 'on_road', True)
            lane = ego_vehicle.lane
            if on_road:
                _, lateral = lane.local_coordinates(ego_vehicle.position)
                half_width = lane.width / 2
                deviation = abs(lateral) / half_width
                if deviation >= 1.0:
                    return rc.get("off_road_penalty", -0.5)
                
                return rc.get("on_road_reward", 0.02)  * (1.0 - deviation)
            else:
                return rc.get("off_road_penalty", -0.5)
        except (AttributeError, TypeError):
            return 0.0

    def _compute_progress_reward(self, ego_vehicle, rc):
        """Compute reward for forward progress."""
        try:
            current_pos = np.array(ego_vehicle.position) if hasattr(ego_vehicle, 'position') else None
            
            if current_pos is not None and self._last_position is not None:
                forward_progress = current_pos[0] - self._last_position[0]
                reward = forward_progress * rc.get("progress_reward_scale", 0.005)
            else:
                reward = 0.0
            
            self._last_position = current_pos
            return reward
        except (AttributeError, TypeError):
            self._last_position = None
            return 0.0

    def _compute_heading_reward(self, ego_vehicle, rc):
        """Compute reward for proper heading alignment."""
        try:
            if not hasattr(ego_vehicle, 'heading') or not hasattr(ego_vehicle, 'lane'):
                return 0.0
            
            lane = ego_vehicle.lane
            if lane is None:
                return 0.0
            
            lane_heading = lane.heading_at(lane.local_coordinates(ego_vehicle.position)[0])
            heading_error = abs(ego_vehicle.heading - lane_heading)
            heading_error = min(heading_error, 2 * np.pi - heading_error)
            
            alignment_reward = (1 - heading_error / np.pi) * rc.get("heading_reward", 0.05)
            return alignment_reward
        except (AttributeError, TypeError):
            return 0.0

    def _compute_lateral_deviation_reward(self, ego_vehicle, rc):
        """
        Reward (or mild penalty) based on how far the ego vehicle is from the lane center.
        
        Uses lane.local_coordinates to get signed lateral offset:
        - Near the center -> positive reward (up to lane_centering_reward).
        - Outside lane bounds -> penalty that grows linearly past the edge.

        - Return the lateral offset for later usage
        """
        try:
            lane = getattr(ego_vehicle, "lane", None)
            if lane is None or not hasattr(lane, "local_coordinates"):
                return 0.0

            _, lateral_offset = lane.local_coordinates(ego_vehicle.position)

            # Estimate lane half-width; fall back to typical 4m if unavailable.
            if hasattr(lane, "width_at"):
                lane_width = float(lane.width_at(0))
            else:
                lane_width = float(getattr(lane, "width", 4.0))
            half_width = max(1e-3, lane_width / 2.0)

            center_weight = rc.get("lane_centering_reward", 0.0)
            if center_weight == 0.0:
                return 0.0, 0.0

            normalized_offset = abs(lateral_offset) / half_width  # 0 at center, 1 at edge
            reward = center_weight * (1.0 - normalized_offset)
            if normalized_offset > 1.0:
                # Outside the lane: dampen further based on how far out we are.
                reward = center_weight * (1.0 - normalized_offset * 1.5)

            return float(np.clip(reward, -center_weight, center_weight)), lateral_offset
        except Exception:
            return 0.0, 0.0

    def _update_comfort_index(self, action, ego_vehicle, dt=0.5):
        # Compute per-step level
        acc, steer = action

        # Longitudinal jerk
        jerk = abs(acc - self.prev_acc) / dt
        self.jerk_sum += jerk

        # Steering rate
        steer_rate = abs(steer - self.prev_steer) / dt
        self.steer_rate_sum += steer_rate

        # Lateral stability
        lane = ego_vehicle.lane
        if lane is not None:
            _, lateral = lane.local_coordinates(ego_vehicle.position)
            self.lat_offsets.append(lateral)

        self.prev_acc = acc
        self.prev_steer = steer
        self.comfort_steps += 1

    def _compute_comfort_index(self):
        # Compute Episode Level
        if self.comfort_steps == 0:
            return 1.0

        mean_jerk = self.jerk_sum / self.comfort_steps
        mean_steer_rate = self.steer_rate_sum / self.comfort_steps
        # Normalize lateral offsets by lane width
        if self.lat_offsets:
            lane = self.unwrapped.vehicle.lane
            half_width = lane.width / 2 if lane else 1.0
            norm_lat = np.array(self.lat_offsets) / half_width
            lat_var = np.var(norm_lat)
        else:
            lat_var = 0.0
        # Tunable weights (highway-env friendly)
        alpha = 1.0    # jerk
        beta = 0.5    # steering
        gamma = 0.05  # lateral oscillation

        discomfort = (
            alpha * mean_jerk +
            beta * mean_steer_rate +
            gamma * lat_var
        )

        # Bounded (0, 1]
        return float(np.exp(-discomfort))

    # =========================================================================
    # Gymnasium Interface
    # =========================================================================

    @property
    def observation_space(self):
        """Return observation space compatible with DreamerV3."""
        if self._obs_type in ("image", "grayscale"):
            channels = 1 if self._obs_type == "grayscale" else 3
            # _size is (width, height), but image arrays are (height, width, channels)
            img_shape = (self._size[1], self._size[0], channels)
            spaces = {
                "image": gymnasium.spaces.Box(0, 255, img_shape, dtype=np.uint8),
                "is_first": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_last": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_terminal": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
            }
        else:
            obs_shape = self._env.observation_space.shape
            spaces = {
                "kinematics": gymnasium.spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32),
                "is_first": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_last": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
                "is_terminal": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
            }
        return gymnasium.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Return action space."""
        gym_space = self._env.action_space
        
        if isinstance(gym_space, gymnasium.spaces.Discrete):
            space = gymnasium.spaces.Discrete(gym_space.n)
            space.discrete = True
        elif isinstance(gym_space, gymnasium.spaces.Box):
            space = gymnasium.spaces.Box(
                low=gym_space.low.astype(np.float32),
                high=gym_space.high.astype(np.float32),
                dtype=np.float32
            )
        else:
            raise NotImplementedError(f"Action space {type(gym_space)} not supported")
        
        return space

    def step(self, action):
        """Execute action and return observation, reward, terminated, truncated, info."""
        # Handle action based on action space type
        if isinstance(self._env.action_space, gymnasium.spaces.Discrete):
            # Discrete: handle one-hot encoded actions
            if hasattr(action, 'shape') and len(action.shape) >= 1 and action.shape[0] > 1:
                action = np.argmax(action)
            elif isinstance(action, np.ndarray):
                action = action.item() if action.size == 1 else int(action)
        else:
            # Continuous: ensure action is a numpy array with correct shape
            if not isinstance(action, np.ndarray):
                action = np.array(action, dtype=np.float32)
            action = action.flatten().astype(np.float32)
        
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self._action_repeat):
            obs, reward, terminated, truncated, info = self._env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        done = terminated or truncated
        self._done = done
        
        # Apply reward shaping
        shaped_reward = self._compute_shaped_reward(total_reward, info, terminated, truncated)

        # Attach predicted/true positions for minADE (best-effort fallback).
        try:
            vehicle = getattr(self._env.unwrapped, "vehicle", None)
            if vehicle is not None and hasattr(vehicle, "position"):
                pos = np.array(vehicle.position, dtype=np.float32)
                vel = np.array(getattr(vehicle, "velocity", np.zeros(2)), dtype=np.float32)
                heading = float(getattr(vehicle, "heading", 0.0))
                dt = 1.0 / float(getattr(self._env.unwrapped, "policy_frequency", 15))
                speed = np.linalg.norm(vel)
                pred_offset = speed * dt * np.array([np.cos(heading), np.sin(heading)], dtype=np.float32)
                info["true_future_position"] = pos
                info["predicted_position"] = pos + pred_offset
        except Exception:
            pass
        
        info["comfort_index"] = None
        try:
            vehicle = getattr(self._env.unwrapped, "vehicle", None)
            self._update_comfort_index(action, vehicle)
            info["comfort_index"] = self._compute_comfort_index()
        except Exception:
            pass

        # Attach ego position and lane center for lateral deviation (signed when available).
        try:
            vehicle = getattr(self._env.unwrapped, "vehicle", None)
            if vehicle is not None and hasattr(vehicle, "position"):
                ego_pos = np.array(vehicle.position, dtype=np.float32)
                info["ego_position"] = ego_pos
                lane = getattr(vehicle, "lane", None)
                if lane is not None and hasattr(lane, "local_coordinates"):
                    _, lateral_offset = lane.local_coordinates(ego_pos)
                    info["lateral_offset_signed"] = float(lateral_offset)
                    lane_width = float(getattr(lane, "width", 4.0))
                    half_width = max(1e-3, lane_width / 2.0)
                    info["lateral_offset_normalized"] = float(min(2.0, abs(lateral_offset) / half_width))
        except Exception:
            pass
        
        # Add the flag off_road for each step (always report actual state on step 0).
        info["off_road"] = False
        info["off_road_entered"] = False
        try:
            vehicle = getattr(self._env.unwrapped, "vehicle", None)
            is_off_road = self._is_off_road(vehicle) if vehicle is not None else True
            info["off_road"] = bool(is_off_road)
            info["off_road_entered"] = bool(self._just_went_off_road)
        except Exception:
            pass

        # Process observation
        processed_obs = self._process_obs(obs, is_first=False, is_last=done, is_terminal=terminated)
        
        # adding the off-road count for each step
        try:
            info["off_road_count"] = self.off_road_count
        except Exception:
            pass
        # Add info
        if "discount" not in info:
            info["discount"] = np.array(0.0 if terminated else 1.0, dtype=np.float32)
        info["original_reward"] = total_reward
        info["shaped_reward"] = shaped_reward
        
        
        return processed_obs, shaped_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            obs, info = self._env.reset(seed=seed, options=options)
        else:
            obs, info = self._env.reset(options=options)
        
        self._done = False
        self._reset_tracking_variables()
        
        return self._process_obs(obs, is_first=True, is_last=False, is_terminal=False), info

    def _process_obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        """Process observation to match DreamerV3 format."""
        # Convert boolean flags to float32 for consistency with observation space
        is_first = np.float32(is_first)
        is_last = np.float32(is_last)
        is_terminal = np.float32(is_terminal)
        
        if self._obs_type in ("image", "grayscale"):
            # Prefer direct observation pixels; avoid on-screen render in headless runs.
            image = None
            if isinstance(obs, dict) and "image" in obs:
                image = obs["image"]
            elif isinstance(obs, np.ndarray):
                image = obs

            if image is None:
                # Fallback to blank frame if no pixels are provided.
                w, h = self._size
                image = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Ensure (H, W, C) format
            if image.ndim == 2:
                image = image[:, :, None]
            if image.shape[2] == 1:
                image = np.repeat(image, 3, axis=2)
            
            # For grayscale, convert RGB to single channel
            if self._obs_type == "grayscale":
                if image.ndim == 3 and image.shape[2] == 3:
                    # RGB to grayscale using standard weights
                    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
                    image = image[:, :, None]
            elif image.ndim == 3 and image.shape[2] == 1:
                # Ensure RGB input for downstream encoders
                image = np.repeat(image, 3, axis=2)
            
            # Resize if needed
            if image.shape[0] != self._size[1] or image.shape[1] != self._size[0]:
                image = self._resize_image(image)
                if image.ndim == 2:
                    image = image[:, :, None]
            
            return {
                "image": image.astype(np.uint8),
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
            }
        else:
            # Kinematics - keep 2D shape (vehicles, features) to match observation space
            kinematics = np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs.astype(np.float32)
            # Don't flatten - keep original shape to match observation_space
            return {
                "kinematics": kinematics,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
            }

    def _is_off_road(self, ego_vehicle):
        lane = ego_vehicle.lane
        if lane is None:
            return True

        _, lateral = lane.local_coordinates(ego_vehicle.position)
        return abs(lateral) > lane.width / 2

    def _update_off_road_count(self, ego_vehicle):
        is_off_road = self._is_off_road(ego_vehicle)
        # Count only on-road → off-road transition
        self._just_went_off_road = bool(is_off_road and self._was_on_road)
        if self._just_went_off_road:
            self.off_road_count += 1

        self._was_on_road = not is_off_road
        
    def _resize_image(self, image):
        """Resize image to target size."""
        try:
            import cv2
            return cv2.resize(image, self._size, interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize(self._size, PILImage.BILINEAR)
            return np.array(pil_img)

    def render(self, mode="rgb_array"):
        """Render the environment."""
        return self._env.render(mode=self._render_mode)

    def close(self):
        """Close the environment."""
        return self._env.close()

    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            self._env.reset(seed=seed)

class HighwayEnvKinematics(HighwayEnv):
    """Highway environment using kinematics observations (vector-based)."""
    
    def __init__(
        self,
        name,
        action_repeat=1,
        vehicles_count=5,
        features=None,
        action_type="discrete",
        seed=None,
        use_reward_shaping=True,
        reward_config=None,
    ):
        """
        Args:
            name: Environment name
            action_repeat: Number of times to repeat each action
            vehicles_count: Number of vehicles to observe
            features: List of features to observe per vehicle
            action_type: 'discrete' or 'continuous'
            seed: Random seed
            use_reward_shaping: Whether to use advanced reward shaping (default True)
            reward_config: Custom reward configuration dict
        """
        self._kin_vehicles_count = vehicles_count
        self._features = features or ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
        
        super().__init__(
            name=name,
            action_repeat=action_repeat,
            obs_type="kinematics",
            action_type=action_type,
            seed=seed,
            use_reward_shaping=use_reward_shaping,
            reward_config=reward_config,
        )

    def _get_observation_config(self):
        """Override to use custom kinematics config."""
        return {
            "observation": {
                "type": "Kinematics",
                "vehicles_count": self._kin_vehicles_count,
                "features": self._features,
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20]
                },
                "absolute": False,
                "order": "sorted",
                "normalize": True,
            }
        }

    @property
    def observation_space(self):
        """Return observation space for kinematics (flattened 1D vector)."""
        # Use flattened 1D shape for compatibility with MLP encoders
        flat_size = self._kin_vehicles_count * len(self._features)
        spaces = {
            "vector": gymnasium.spaces.Box(-1.0, 1.0, (flat_size,), dtype=np.float32),
            "is_first": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
            "is_last": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
            "is_terminal": gymnasium.spaces.Box(0, 1, (), dtype=np.float32),
        }
        return gymnasium.spaces.Dict(spaces)

    def _process_obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        """Process kinematics observation to flattened 1D vector."""
        # Convert flags to float32 for consistency
        is_first = np.float32(is_first)
        is_last = np.float32(is_last)
        is_terminal = np.float32(is_terminal)
        
        # Convert to numpy and flatten to 1D
        vector = np.array(obs, dtype=np.float32) if not isinstance(obs, np.ndarray) else obs.astype(np.float32)
        vector = vector.flatten()  # Always flatten to match observation_space
        
        return {
            "vector": vector,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
