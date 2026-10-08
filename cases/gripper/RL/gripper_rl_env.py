"""Gym environment for task-driven soft-gripper morphology optimization.

Defines design actions, task sampling, observations and rewards for PPO.
Each episode evaluates a grasp using the surrogate-driven task simulator."""

import os
import time
from pathlib import Path

# ENV VARS (must be set before TF / MKL loads)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# PATH SETUP (robust against CWD / spawn)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(THIS_DIR))

# IMPORTS
import gym
from gym import spaces
import numpy as np

from gripper_simulation_rl import gripper_simulator


class GripperEnv(gym.Env):
    """
    Task-driven design environment for soft gripper optimization using RL.
    Supports two modes:
      - mode='compact': returns compressed observation for RL training
      - mode='full': returns full traces for analysis/debug
    """

    action_continuous_low = np.array([5, 10, 0, 0, 2, 6, 8])
    action_continuous_high = np.array([15, 15, 10, np.pi/6, 4, 8, 12])
    
    
    def __init__(self, object_types=None, mode="compact", seed=None, verbose=False):
        super().__init__()
        assert mode in ["compact", "full"], "mode must be 'compact' or 'full'"
        self.verbose = verbose
        self.mode = mode
        
        # Object & Task Randomization
        if object_types is not None:
            self.object_types = object_types
        else:
            self.object_types = ["egg", "cylinder"]
            
        self.grd_bounds = {  # object-specific grasp_radius_distance bounds (mm)
            "egg": (-16.0, 0.0),
            "cylinder": (-4.0, 4.0),
        }
        self.weight_choices = {
            # "egg": [4, 5, 6],
            "egg": [5, 6, 7],
            "cylinder": [2, 3, 4],
        }
        self.P_ref = 15.0  # reference pressure (kPa)
        self.lambda_p = 0.5  # pressure penalty weight


        # Action space
        self.n_bins = [5, 6, 6, 6, 5, 5, 5]  # bins per continuous dimension
        discrete_actions_per_dim = self.n_bins
        action_module_number = 6  # 5–10
        action_actuator_number = 3  # 2–4
        self.action_space = spaces.MultiDiscrete(discrete_actions_per_dim + [action_module_number, action_actuator_number])

        
        # Observation space
        if self.mode == "compact":

            obs_dim = 7 + len(self.object_types) + 1
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        else:
            self.max_n_steps = 9600
            self.max_actuator_number = 4
            self.max_module_number = 10
            self.observation_space = spaces.Dict({
                "grasping_mode": spaces.Discrete(6, start=0),
                "object_position": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_n_steps,), dtype=np.float32),
                "contact_normal": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_actuator_number, self.max_module_number, self.max_n_steps), dtype=np.float32),
                "contact_friction": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_actuator_number, self.max_module_number, self.max_n_steps), dtype=np.float32),
                "object_properties": spaces.Box(low=0, high=1, shape=(len(self.object_types),), dtype=np.float32),
            })


        # Misc
        self.current_object_type = None
        self.current_weight_num = None
        self.seed(seed)
        self.w_min = min(min(v) for v in self.weight_choices.values())
        self.w_max = max(max(v) for v in self.weight_choices.values())
        
    # STEP FUNCTION
    def step(self, action):
        # Decode actions
        decoded = self._decode_action(action)
        (init_grasp_height, maximum_pressure, grasp_radius_distance,
         inclination_angle, inner_radius, average_radius, module_length,
         module_number, actuator_number) = decoded

        wall_thickness = 1.5
        add_noise = True
        extra_weight = True

        sim_start_time = time.time()
        # Run simulation
        sim_output = gripper_simulator(
            init_grasp_height=init_grasp_height,
            maximum_pressure=maximum_pressure,
            grasp_radius_distance=grasp_radius_distance,
            inclination_angle=inclination_angle,
            inner_radius=inner_radius,
            wall_thickness=wall_thickness,
            average_radius=average_radius,
            module_length=module_length,
            module_number=module_number,
            actuator_number=actuator_number,
            object_type=self.current_object_type,
            extra_weight=extra_weight,
            weight_num=self.current_weight_num,
            add_noise=add_noise,
            render=False,
            plot=False,
            seed=None,
            return_traces=(self.mode == "full"),
            verbose=self.verbose,
        )
        sim_end_time = time.time()
        if isinstance(sim_output, tuple):
            # full mode
            obs_compact, obj_pos, cn, cf = sim_output
        else:
            # compact mode
            obs_compact = sim_output
            obj_pos, cn, cf = None, None, None
            
        # Extract observation
        if self.mode == "compact":
            grasping_mode = obs_compact["grasping_mode"]
            obj_z_max = obs_compact["obj_z_max"]
            obj_z_end = obs_compact["obj_z_end"]
            obj_v_lift_mean = obs_compact["obj_v_lift_mean"]
            cn_sum = obs_compact["contact_normal_sum"]
            cf_sum = obs_compact["contact_friction_sum"]
            c_link_eff = obs_compact["contact_link_eff"]
            slip = obs_compact["slip_ratio"]
            
            obj_type_enc = self._one_hot_encode(self.current_object_type, self.object_types)
            obs_vec = np.array([obj_z_max, obj_z_end, obj_v_lift_mean, cn_sum, cf_sum, slip, grasping_mode], dtype=np.float32)
            observation = np.concatenate([obs_vec, obj_type_enc, np.array([self.weight_value], dtype=np.float32)])
        else:
            max_actuators = 4
            max_modules = 10
            max_steps = 9600

            padded_cn = self._pad_matrix(cn, max_actuators, max_modules, max_steps)
            padded_cf = self._pad_matrix(cf, max_actuators, max_modules, max_steps)
            
            
            observation = {
                "grasping_mode": obs_compact["grasping_mode"],
                "object_position": self._pad_array(obj_pos, max_steps),
                "contact_normal": padded_cn,
                "contact_friction": padded_cf,
                "object_properties": np.concatenate([
                    self._one_hot_encode(self.current_object_type, self.object_types),
                    np.array([self.weight_value], dtype=np.float32)
                ])
            }

        # Compute reward
        reward = self._compute_reward(obs_compact, maximum_pressure)

        # Done after one grasp simulation
        done = True
        info = {
            "object_type": self.current_object_type,
            "weight_num": self.current_weight_num,
            "init_grasp_height": decoded[0],
            "maximum_pressure": decoded[1],
            "grasp_radius_distance": round(decoded[2], 2),
            "inclination_angle": round(decoded[3]/np.pi*180, 2),
            "inner_radius": decoded[4],
            "average_radius": decoded[5],
            "module_length": decoded[6],
            "module_number": decoded[7],
            "actuator_number": decoded[8],
            # "decoded_action": decoded
        }
        
        
        if self.verbose:
            print(f"[observation] {self.current_object_type} (w={self.current_weight_num}), mode={self.mode} → "
                  f"sim_time={sim_end_time - sim_start_time:.3f}s, reward={reward:.3f}, "
                  f"obj_z_max={obs_compact['obj_z_max']:.3f}, obj_z_end={obs_compact['obj_z_end']:.3f}, slip_ratio={obs_compact['slip_ratio']:.3f}, contact_link_eff={obs_compact['contact_link_eff']:.3f}, "
                  f"pressure={maximum_pressure:.1f}, grasping_mode={obs_compact['grasping_mode']}")
            print("[Done] ", done)
            print("[Info]", info)
        print("===============================================")
        return observation, reward, done, info
    
    # RESET FUNCTION
    def reset(self, **kwargs):
        self.current_object_type = np.random.choice(self.object_types)
        self.current_weight_num = np.random.choice(self.weight_choices[self.current_object_type])
        self.weight_value = float(self.current_weight_num)
        
        if self.verbose:
            print(f"[RESET] object={self.current_object_type}, weight={self.current_weight_num}")
        
        if self.mode == "compact":
            # obs: [obj_z_max, obj_z_end, obj_v_lift_mean, cn_sum, cf_sum, slip, grasping_mode] + onehot(obj) + weight_norm
            obs = np.zeros(7 + len(self.object_types) + 1, dtype=np.float32)
            obs[-(len(self.object_types) + 1):-1] = self._one_hot_encode(self.current_object_type, self.object_types)
            obs[-1] = self.weight_value  
        else:
            max_actuators = 4
            max_modules = 10
            max_steps = 9600
            obs = {
                "grasping_mode": 0,
                "object_position": np.zeros(max_steps, dtype=np.float32),
                "contact_normal": np.zeros((max_actuators, max_modules, max_steps), dtype=np.float32),
                "contact_friction": np.zeros((max_actuators, max_modules, max_steps), dtype=np.float32),
                "object_properties": np.concatenate([
                    self._one_hot_encode(self.current_object_type, self.object_types),
                    np.array([self.weight_value], dtype=np.float32)  
                ])
            }
        return obs

    # HELPER FUNCTIONS
    
    
    def _decode_action(self, action):
        """Convert discrete bins to continuous design parameters."""
        n_cont = 7
        discretized = action[:n_cont]
        decoded = []
        for i, disc_val in enumerate(discretized):
            low, high = self.action_continuous_low[i], self.action_continuous_high[i]
            # special bound for grasp_radius_distance
            if i == 2:
                low, high = self.grd_bounds[self.current_object_type]
            step = (high - low) / (self.n_bins[i] - 1)
            decoded.append(low + step * disc_val)

        # constraint: R - r >= l / 4
        if not hasattr(self, "_constraint_hits"):
            self._constraint_hits = 0
        if decoded[5] - decoded[4] < decoded[6] / 4:
            self._constraint_hits += 1
            R_min = decoded[6] / 4 + decoded[4]
            R_max = self.action_continuous_high[5]
            step = (R_max - R_min) / (self.n_bins[5] - 1)
            decoded[5] = R_min + step * discretized[5]
            
        if self.verbose:
            print(f"[constraint] hit #{self._constraint_hits}: "
                f"r={decoded[4]:.2f}, l={decoded[6]:.2f}, R={decoded[5]:.2f}")


        module_number = action[7] + 5
        actuator_number = action[8] + 2
        decoded += [module_number, actuator_number]
        return decoded


    def _compute_reward(self, obs_compact, maximum_pressure):
        """Reward based on grasp success and stability, with task-aware scaling."""
        gm = obs_compact["grasping_mode"]
        z_max = obs_compact["obj_z_max"]
        z_end = obs_compact["obj_z_end"]
        v_lift_mean = obs_compact["obj_v_lift_mean"]
        cn_sum = obs_compact["contact_normal_sum"]
        cf_sum = obs_compact["contact_friction_sum"]
        c_link_eff = obs_compact["contact_link_eff"]
        slip = obs_compact["slip_ratio"]

        # 🧩 Original reward components
        z_reward = min(1.0, max(0.0, z_end / 60))             # normalized lift
        contact_eff = cf_sum / (cn_sum + 1e-9)                # friction/contact ratio
        base_reward = {0: 0, 1: -1, 2: 0.5, 3: 1, 4: 0, 5: 2}[gm]

        # Pressure penalty
        pressure_penalty = self.lambda_p * (maximum_pressure / self.P_ref)

        # Base reward (task-independent)
        reward = base_reward + 2.0 * z_reward + 0.5 * contact_eff - 0.1 * slip - pressure_penalty

        # --- Actuator complexity light cost ---
        # Penalize larger total actuator count (actuator_number × module_number)
        actuator_num = obs_compact.get("actuator_number", None)
        module_num = obs_compact.get("module_number", None)

        if actuator_num is not None and module_num is not None:
            total_actuators = actuator_num * module_num
            reward -= 0.005 * total_actuators 

        # Task-driven scaling only
        if self.current_object_type == "egg":
            # prefer distributed contact (power grasp)
            reward += 0.2 * c_link_eff
            task_scale = 1.0 + 0.05 * (self.current_weight_num - 5)
        elif self.current_object_type == "cylinder":
            # prefer localized contact (precision grasp)
            reward += 0.2 * (1.0 - c_link_eff)
            task_scale = 1.0 + 0.07 * (self.current_weight_num - 2)
        else:
            task_scale = 1.0

        reward *= task_scale
        return float(reward)

    
    def _one_hot_encode(self, value, categories):
        enc = np.zeros(len(categories), dtype=np.float32)
        enc[categories.index(value)] = 1.0
        return enc

    def _pad_array(self, array, length):
        padded = np.zeros(length, dtype=np.float32)
        padded[:min(len(array), length)] = array[:min(len(array), length)]
        return padded

    def _pad_matrix(self, matrix, d1, d2, d3):
        padded = np.zeros((d1, d2, d3), dtype=np.float32)
        shape = matrix.shape
        padded[:shape[0], :shape[1], :shape[2]] = matrix
        return padded

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]


if __name__ == "__main__":
    env = GripperEnv(mode="compact", verbose=True)
    obs = env.reset()
    print("Initial obs:", obs)

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("Next obs:", obs)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)
    

    print("\nSwitching to full mode...")
    env_full = GripperEnv(mode="full", verbose=True)
    obs_full = env_full.reset()
    print("Initial keys in full obs:", list(obs_full.keys()))
    action = env_full.action_space.sample()
    obs_full, reward, done, info = env_full.step(action)
    print("Reward:", reward)
    print("Full obs keys:", list(obs_full.keys()))
