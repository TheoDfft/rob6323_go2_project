# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfWaveTerrainCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

# === ADDED Part 2: PD controller ===
# add this import:
from isaaclab.actuators import ImplicitActuatorCfg

@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 20.0
    # - spaces definition
    action_scale = 0.25
    action_space = 12
    observation_space = 48 + 4  # === MODIFIED Part 4: Added 4 for clock inputs ===
    state_space = 0
    debug_vis = True
    # === ADDED Part 3: Early termination ===
    base_height_min = 0.20  # Terminate if base is lower than 20cm

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    # === MODIFIED: Wave terrain for locomotion challenge ===
    # Reference settings from Go2Terrain.yaml: 8m x 8m tiles, 10 levels, 20 terrains
    # Robot length ~0.7m, peak-to-peak wavelength >= 1.4m (2x robot length)
    # With 8m terrain and 5 waves: wavelength = 8/5 = 1.6m per wave (meets requirement)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(40.0, 40.0),  # 8m x 8m per tile (from reference)
            border_width=20.0,  # Flat spawn border
            num_rows=1,  # numLevels from reference
            num_cols=1,  # numTerrains from reference
            horizontal_scale=0.1,  # Resolution: 10cm per height sample
            vertical_scale=0.005,  # Height scale factor
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "waves": HfWaveTerrainCfg(
                    proportion=1.0,  # 100% wave terrain
                    amplitude_range=(0.2, 0.2),  # Wave height: 2-6cm amplitude
                    num_waves=2,  # 5 waves per 8m = 1.6m wavelength (> 1.4m requirement)
                    border_width=0.5,
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    # === END MODIFIED ===

    # robot(s)
    # robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # === ADDED Part 2: PD control gains ===
    Kp = 20.0  # Proportional gain
    Kd = 0.5   # Derivative gain
    torque_limits = 100.0  # Max torque

    # === MODIFIED Part 2: Disable implicit actuator PD, use custom PD ===
    # Update robot_cfg
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    # "base_legs" is an arbitrary key we use to group these actuators
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # CRITICAL: Set to 0 to disable implicit P-gain
        damping=0.0,    # CRITICAL: Set to 0 to disable implicit D-gain
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    # === ADDED Part 1: Action rate penalty ===
    action_rate_reward_scale = -0.1
    # === ADDED Part 4: Raibert heuristic ===
    raibert_heuristic_reward_scale = -10.0
    # === ADDED: Part 5 rewards ===
    orient_reward_scale = 0.0  # Penalize non-flat orientation
    lin_vel_z_reward_scale = 0.0  # Penalize vertical bouncing
    dof_vel_reward_scale = -0.0001  # Penalize high joint velocities
    ang_vel_xy_reward_scale = -0.001  # Penalize roll/pitch angular velocity
    # === ADDED: Part 6: Foot clearance ===
    feet_clearance_reward_scale = -30.0  # Penalize low foot height during swing
    # === ADDED: Part 6: Tracking contacts shaped force ===
    tracking_contacts_shaped_force_reward_scale = 4.0  # Reward matching gait contact plan

