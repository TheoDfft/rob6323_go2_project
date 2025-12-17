# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# BIPEDAL STANCE IMPLEMENTATION
# Reference: go2_bipedal.py line 1522 (reward), lines 1894-1901 (terminations)

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
import isaaclab.utils.math as math_utils

from .rob6323_go2_env_cfg import Rob6323Go2EnvCfg


class Rob6323Go2Env(DirectRLEnv):
    cfg: Rob6323Go2EnvCfg

    def __init__(self, cfg: Rob6323Go2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # === BIPEDAL: Key vectors for upright posture ===
        # forward_vec: robot's forward direction in body frame
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        # upright_vec: target upright direction (slightly tilted forward)
        self.upright_vec = torch.tensor([0.2, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)

        # === BIPEDAL: Front/Rear feet indices ===
        self.fidx = [0, 1]      # Front feet: FL, FR
        self.rear_fidx = [2, 3]  # Rear feet: RL, RR

        # Logging - bipedal reward terms
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "upright",
                "lift_up", 
                "stand_air",
                "lin_vel_xy",
                "feet_clearance",
                "tracking_contacts",
                "torque_limits",
                "ang_vel_z",
                "action_rate",
                "dof_vel",
            ]
        }

        # Action history for rate penalization (keep 3 frames: current, t-1, t-2)
        self.last_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), 3, 
            dtype=torch.float, device=self.device, requires_grad=False
        )

        # === BIPEDAL: DOF position history for abrupt_change_condition ===
        self.last_dof_pos = torch.zeros(
            self.num_envs, 12, 1,  # Just need previous frame
            dtype=torch.float, device=self.device, requires_grad=False
        )

        # === BIPEDAL: Store computed torques for torque_limits reward ===
        self.computed_torques = torch.zeros(self.num_envs, 12, device=self.device)

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")

        # add handle for debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # PD control parameters
        self.Kp = torch.tensor([cfg.Kp] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.Kd = torch.tensor([cfg.Kd] * 12, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.motor_offsets = torch.zeros(self.num_envs, 12, device=self.device)
        self.torque_limits_value = cfg.torque_limits

        # Feet body indices
        self._feet_ids = []
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        for name in foot_names:
            id_list, _ = self.robot.find_bodies(name)
            self._feet_ids.append(id_list[0])

        # Feet indices in CONTACT SENSOR (different from robot body indices!)
        self._feet_ids_sensor = []
        for name in foot_names:
            id_list, _ = self._contact_sensor.find_bodies(name)
            self._feet_ids_sensor.append(id_list[0])

        # Gait variables (for front feet only in bipedal)
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        # === BIPEDAL: Joint position limits for position_protect termination ===
        # Go2 joint limits (approximate values in radians)
        self.dof_pos_limits = torch.zeros(12, 2, device=self.device)
        # Hip joints: [-1.047, 1.047] rad
        self.dof_pos_limits[[0, 3, 6, 9], 0] = -1.047
        self.dof_pos_limits[[0, 3, 6, 9], 1] = 1.047
        # Thigh joints: [-1.5, 3.4] rad
        self.dof_pos_limits[[1, 4, 7, 10], 0] = -1.5
        self.dof_pos_limits[[1, 4, 7, 10], 1] = 3.4
        # Calf joints: [-2.7, -0.83] rad
        self.dof_pos_limits[[2, 5, 8, 11], 0] = -2.7
        self.dof_pos_limits[[2, 5, 8, 11], 1] = -0.83

        # === BIPEDAL: Max DOF change threshold for abrupt_change_condition ===
        self.max_dof_change = 0.75  # radians

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        # add ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = actions.clone()
        # Compute desired joint positions from policy actions
        self.desired_joint_pos = (
            self.cfg.action_scale * self._actions
            + self.robot.data.default_joint_pos
        )

    def _apply_action(self) -> None:
        # Compute PD torques
        self.computed_torques = torch.clip(
            (
                self.Kp * (self.desired_joint_pos - self.robot.data.joint_pos)
                - self.Kd * self.robot.data.joint_vel
            ),
            -self.torque_limits_value,
            self.torque_limits_value,
        )
        # Apply torques to the robot
        self.robot.set_joint_effort_target(self.computed_torques)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        
        # === BIPEDAL: Compute yaw-rotated upright vector (replaces commands) ===
        upright_vec_obs = math_utils.quat_apply_yaw(self.robot.data.root_quat_w, self.upright_vec)
        
        obs = torch.cat(
            [
                self.robot.data.root_lin_vel_b,           # (3,) base linear velocity
                self.robot.data.root_ang_vel_b,           # (3,) base angular velocity
                self.robot.data.projected_gravity_b,       # (3,) projected gravity
                upright_vec_obs,                           # (3,) yaw-rotated upright vector (replaces commands)
                self.robot.data.joint_pos - self.robot.data.default_joint_pos,  # (12,) joint positions
                self.robot.data.joint_vel,                 # (12,) joint velocities
                self._actions,                             # (12,) previous actions
                self.clock_inputs,                         # (4,) gait phase info
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Compute bipedal standing rewards.
        Reference: go2_bipedal.py line 1522
        rew_buf = rew_upright + rew_lift_up + rew_stand_air + rew_lin_vel_xy + 
                  rew_feet_clearance + rew_tracking_contacts_shaped_force + 
                  rew_torque_limits + rew_ang_vel_z + rew_action_rate + rew_dof_vel
        """
        
        # === Phase conditions ===
        allow_contact_steps = self.cfg.allow_contact_steps
        condition_tc = self.episode_length_buf > allow_contact_steps
        condition_early = ~condition_tc  # t <= t_c
        
        # === Compute is_stand indicator ===
        forward = math_utils.quat_rotate(self.robot.data.root_quat_w, self.forward_vec)
        upright_vec = math_utils.quat_apply_yaw(self.robot.data.root_quat_w, self.upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / (torch.norm(upright_vec, dim=-1) + 1e-6)
        is_stand = (cosine_dist > 0.9).float()
        
        # === Height scaling factor ===
        root_height = self.robot.data.root_pos_w[:, 2]
        height_scale = torch.clamp(
            (root_height - self.cfg.scale_factor_low) / (self.cfg.scale_factor_high - self.cfg.scale_factor_low),
            0.0, 1.0
        )
        
        # === Update gait/contact states ===
        self._step_contact_targets()
        
        # =====================================================================
        # REWARD TERMS (from line 1522)
        # =====================================================================
        
        # === 1. rew_upright: Upright posture reward ===
        # k_up * (0.5 * cosine_dist + 0.5)^2
        rew_upright = torch.square(0.5 * cosine_dist + 0.5) * self.cfg.upright_reward_scale
        
        # === 2. rew_lift_up: Height reward ===
        # k_lift * clip((H - H_min) / (H_max - H_min), 0, 1)
        lift_up_ratio = (root_height - self.cfg.lift_up_h_min) / (self.cfg.lift_up_h_max - self.cfg.lift_up_h_min)
        rew_lift_up = torch.clamp(lift_up_ratio, 0.0, 1.0) * self.cfg.lift_up_reward_scale
        
        # === 3. rew_stand_air: Early phase stand-air penalty/reward ===
        # Penalty: -k_air_pen * Σ_{front} max(0, h_f - 0.06) when t < t_c
        # Reward: k_air_rew * Σ_{rear} min(h_f, 0.06) when t < t_c
        foot_heights = self.foot_positions_w[:, :, 2]
        front_heights = foot_heights[:, self.fidx]  # Front feet heights
        rear_heights = foot_heights[:, self.rear_fidx]  # Rear feet heights
        
        stand_air_penalty = torch.sum((front_heights - 0.06).clamp(min=0.0), dim=1) * self.cfg.stand_air_penalty_scale
        stand_air_reward = torch.sum(rear_heights.clamp(max=0.06), dim=1) * self.cfg.stand_air_reward_scale
        rew_stand_air = (stand_air_penalty + stand_air_reward) * condition_early.float()
        
        # === 4. rew_lin_vel_xy: No velocity reward + velocity penalty ===
        # Reward: k_lin * exp(-||v_xy||^2 / δ) * is_stand * height_scale
        # Penalty: -k_vel_pen * ||v_xy||^2 * 1_{t>t_c}
        actual_lin_vel = self.robot.data.root_lin_vel_b[:, :2]
        lin_vel_sq = torch.sum(torch.square(actual_lin_vel), dim=1)
        
        # No velocity reward (when standing)
        lin_vel_reward = torch.exp(-lin_vel_sq / self.cfg.lin_vel_delta) * is_stand * height_scale * self.cfg.lin_vel_reward_scale
        # Velocity penalty (after t_c)
        vel_penalty = lin_vel_sq * condition_tc.float() * self.cfg.vel_penalty_scale
        rew_lin_vel_xy = lin_vel_reward + vel_penalty
        
        # === 5. rew_feet_clearance: Front feet clearance (after t_c) ===
        # -k_clr * 1_{t>t_c} * Σ_{front} (h_f - h_target)^2 * (1 - C_f)
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, self.fidx] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        target_height = 0.08 * phases + 0.02  # 8cm max + 2cm foot radius
        clearance_error = torch.square(target_height - front_heights) * (1 - self.desired_contact_states[:, self.fidx])
        rew_feet_clearance = torch.sum(clearance_error, dim=1) * condition_tc.float() * self.cfg.feet_clearance_reward_scale
        
        # === 6. rew_tracking_contacts_shaped_force: Contact tracking (front feet, when standing) ===
        # -k_cfs * is_stand * Σ_{front} (1 - C_f) * (1 - exp(-||f_f||^2 / 100))
        foot_forces = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_sensor, :], dim=-1)
        front_forces = foot_forces[:, self.fidx]
        tracking_contact_penalty = torch.zeros(self.num_envs, device=self.device)
        for i in range(2):  # Only front feet
            tracking_contact_penalty += -(1 - self.desired_contact_states[:, self.fidx[i]]) * (
                1 - torch.exp(-front_forces[:, i] ** 2 / 100.0)
            )
        rew_tracking_contacts = tracking_contact_penalty * is_stand * self.cfg.tracking_contacts_reward_scale / 2
        
        # === 7. rew_torque_limits: Soft torque limit penalty ===
        # -k_τ * Σ_j max(0, |τ_j| - τ_max * σ_s)
        torque_excess = (torch.abs(self.computed_torques) - self.torque_limits_value * self.cfg.soft_torque_limit).clamp(min=0.0)
        rew_torque_limits = torch.sum(torque_excess, dim=1) * self.cfg.torque_limits_reward_scale
        
        # === 8. rew_ang_vel_z: Angular velocity Z penalty (after t_c) ===
        # -k_ω * ||ω_z||^2 * 1_{t>t_c}
        rew_ang_vel_z = torch.square(self.robot.data.root_ang_vel_b[:, 2]) * condition_tc.float() * self.cfg.ang_vel_z_reward_scale
        
        # === 9. rew_action_rate: Action smoothness penalty ===
        # -k_act * ||a_t - a_{t-1}||^2 (1st order)
        # -k_act * ||(a_{t-2} - a_{t-1}) - (a_{t-1} - a_t)||^2 (2nd order)
        action_diff_1 = torch.sum(torch.square(self._actions - self.last_actions[:, :, 0]), dim=1)
        action_diff_2 = torch.sum(torch.square(
            self._actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]
        ), dim=1)
        rew_action_rate = (action_diff_1 + action_diff_2) * (self.cfg.action_scale ** 2) * self.cfg.action_rate_reward_scale
        
        # Update action history
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self._actions[:]
        
        # === 10. rew_dof_vel: Joint velocity penalty ===
        # -k_q̇ * ||q̇||^2
        rew_dof_vel = torch.sum(torch.square(self.robot.data.joint_vel), dim=1) * self.cfg.dof_vel_reward_scale
        
        # =====================================================================
        # TOTAL REWARD (line 1522)
        # =====================================================================
        rewards = {
            "upright": rew_upright,
            "lift_up": rew_lift_up,
            "stand_air": rew_stand_air,
            "lin_vel_xy": rew_lin_vel_xy,
            "feet_clearance": rew_feet_clearance,
            "tracking_contacts": rew_tracking_contacts,
            "torque_limits": rew_torque_limits,
            "ang_vel_z": rew_ang_vel_z,
            "action_rate": rew_action_rate,
            "dof_vel": rew_dof_vel,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Clip reward to be non-negative (from reference line 1525-1527)
        reward = torch.clip(reward, min=0.0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
            
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute bipedal termination conditions.
        Reference: go2_bipedal.py lines 1894-1901
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        allow_contact_steps = self.cfg.allow_contact_steps
        allow_not_stand_steps = self.cfg.allow_not_stand_steps
        
        # === any_contacts: Base contact termination ===
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        cstr_base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, 
            dim=1
        )
        
        # === stand_air_condition: Front feet too high during early phase ===
        # Terminate if front feet > 6cm during early phase (3 < t <= allow_contact_steps)
        foot_heights = self.foot_positions_w[:, :, 2]
        front_heights = foot_heights[:, self.fidx]
        stand_air_condition = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= allow_contact_steps),
            torch.any(front_heights > 0.06, dim=-1)
        )
        
        # === abrupt_change_condition: Joint positions changed too fast ===
        abrupt_change_condition = torch.logical_and(
            torch.logical_and(self.episode_length_buf > 3, self.episode_length_buf <= allow_contact_steps),
            torch.any(torch.abs(self.robot.data.joint_pos - self.last_dof_pos[:, :, 0]) > self.max_dof_change, dim=-1)
        )
        
        # === position_protect: Joints near limits ===
        joint_pos = self.robot.data.joint_pos
        near_lower = joint_pos < (self.dof_pos_limits[:, 0] + 5.0 / 180.0 * np.pi)
        near_upper = joint_pos > (self.dof_pos_limits[:, 1] - 5.0 / 180.0 * np.pi)
        position_protect = torch.logical_and(
            self.episode_length_buf > 3,
            torch.any(torch.logical_or(near_lower, near_upper), dim=-1)
        )
        
        # === not_stand: Not standing after allow_not_stand_steps ===
        forward = math_utils.quat_rotate(self.robot.data.root_quat_w, self.forward_vec)
        upright_vec = math_utils.quat_apply_yaw(self.robot.data.root_quat_w, self.upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / (torch.norm(upright_vec, dim=-1) + 1e-6)
        is_stand = cosine_dist > 0.9
        not_stand = torch.logical_and(
            self.episode_length_buf > allow_not_stand_steps,
            ~is_stand
        )
        
        # === cstr_base_height_min: Base height too low ===
        base_height = self.robot.data.root_pos_w[:, 2]
        cstr_base_height_min = base_height < self.cfg.base_height_min
        
        # === Combine all termination conditions (lines 1894-1901) ===
        died = cstr_base_contact
        died = died | abrupt_change_condition
        died = died | position_protect
        died = died | not_stand
        died = died | stand_air_condition
        died = died | cstr_base_height_min
        
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        
        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid training spikes
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        
        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        # Reset action history
        self.last_actions[env_ids] = 0.0
        
        # Reset DOF position history
        self.last_dof_pos[env_ids] = self.robot.data.joint_pos[env_ids].unsqueeze(-1)
        
        # Reset gait indices
        self.gait_indices[env_ids] = 0

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        # Visualize upright direction
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # Show forward direction
        forward = math_utils.quat_rotate(self.robot.data.root_quat_w, self.forward_vec)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(forward[:, :2])
        # Show upright target
        upright_vec = math_utils.quat_apply_yaw(self.robot.data.root_quat_w, self.upright_vec)
        upright_scale, upright_quat = self._resolve_xy_velocity_to_arrow(upright_vec[:, :2])
        
        self.goal_vel_visualizer.visualize(base_pos_w, upright_quat, upright_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts XY vector to arrow direction rotation."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    @property
    def foot_positions_w(self) -> torch.Tensor:
        """Returns feet positions in world frame. Shape: (num_envs, 4, 3)"""
        return self.robot.data.body_pos_w[:, self._feet_ids]

    def _step_contact_targets(self):
        """Compute gait phase and desired contact states for front feet."""
        frequencies = 3.0
        phases = 0.5
        offsets = 0.0
        bounds = 0.0
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        
        self.gait_indices = torch.remainder(self.gait_indices + self.step_dt * frequencies, 1.0)

        foot_indices = [
            self.gait_indices + phases + offsets + bounds,  # FL
            self.gait_indices + offsets,                     # FR
            self.gait_indices + bounds,                      # RL
            self.gait_indices + phases                       # RR
        ]

        self.foot_indices = torch.remainder(
            torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 
            1.0
        )

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations
            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # Von Mises distribution for smooth contact transitions
        kappa = 0.07
        smoothing_cdf = torch.distributions.normal.Normal(0, kappa).cdf

        for i, foot_idx in enumerate(foot_indices):
            smoothing_multiplier = (
                smoothing_cdf(torch.remainder(foot_idx, 1.0)) * 
                (1 - smoothing_cdf(torch.remainder(foot_idx, 1.0) - 0.5)) +
                smoothing_cdf(torch.remainder(foot_idx, 1.0) - 1) * 
                (1 - smoothing_cdf(torch.remainder(foot_idx, 1.0) - 0.5 - 1))
            )
            self.desired_contact_states[:, i] = smoothing_multiplier

        # Update last_dof_pos for abrupt_change_condition
        self.last_dof_pos[:, :, 0] = self.robot.data.joint_pos.clone()
