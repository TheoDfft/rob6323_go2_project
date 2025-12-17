# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.actuators import ImplicitActuatorCfg


@configclass
class Rob6323Go2EnvCfg(DirectRLEnvCfg):
    # === BIPEDAL STANDING CONFIGURATION ===
    
    # env
    decimation = 4
    episode_length_s = 20.0
    
    # Spaces definition
    # Observation: 3 (lin_vel) + 3 (ang_vel) + 3 (gravity) + 3 (upright_vec) + 12 (joint_pos) + 12 (joint_vel) + 12 (actions) + 4 (clock) = 52
    action_scale = 0.25
    action_space = 12
    observation_space = 52
    state_space = 0
    debug_vis = True

    # === BIPEDAL: Phase control thresholds ===
    allow_contact_steps = 30   # t_c: when to start applying standing rewards
    allow_not_stand_steps = 60  # when to require standing posture
    
    # === BIPEDAL: Lift-up height thresholds ===
    lift_up_h_min = 0.27  # H_min for lift-up reward
    lift_up_h_max = 0.55  # H_max for lift-up reward
    
    # === BIPEDAL: Base height termination threshold ===
    base_height_min = 0.15  # Terminate if base is lower than 15cm

    # === BIPEDAL: Scaling factors for no-velocity reward ===
    scale_factor_low = 0.33
    scale_factor_high = 0.45
    lin_vel_delta = 0.25  # δ for exp kernel

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
    
    # Flat terrain for bipedal standing (simpler than walking)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
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

    # === PD control gains (reused from quadruped) ===
    Kp = 20.0  # Proportional gain
    Kd = 0.5   # Derivative gain
    torque_limits = 23.5  # Max torque (from Go2 specs)
    
    # === BIPEDAL: Soft torque limit for reward ===
    soft_torque_limit = 0.5  # σ_s: fraction of max torque before penalty

    # Robot configuration with custom PD
    robot_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot_cfg.actuators["base_legs"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
        effort_limit=23.5,
        velocity_limit=30.0,
        stiffness=0.0,  # Disable implicit P-gain (use custom PD)
        damping=0.0,    # Disable implicit D-gain (use custom PD)
    )
    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )
    
    # Visualization markers
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)

    # === BIPEDAL REWARD SCALES (from Paper Table 3) ===
    # Rewards (positive = good)
    upright_reward_scale = 1.0           # k_up: Upright posture reward
    lift_up_reward_scale = 0.5           # k_lift: Height increase reward
    stand_air_reward_scale = 5.0         # k_air_rew: Reward rear feet grounded early
    lin_vel_reward_scale = 1.0           # k_lin: No velocity reward when standing
    
    # Penalties (negative = bad)
    stand_air_penalty_scale = -40.0      # k_air_pen: Penalize front feet > 6cm early
    vel_penalty_scale = -0.4             # k_vel_pen: Velocity penalty when standing
    ang_vel_z_reward_scale = -0.1        # k_ω: Angular velocity Z penalty
    action_rate_reward_scale = -0.03     # k_act: Action rate penalty (1st + 2nd order)
    dof_vel_reward_scale = -0.0001       # k_q̇: Joint velocity penalty
    torque_limits_reward_scale = -0.01   # k_τ: Torque limits penalty
    feet_clearance_reward_scale = -300.0  # k_clr: Front feet clearance penalty
    tracking_contacts_reward_scale = -1.0   # k_cfs: Contact tracking (front feet only)
