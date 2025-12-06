# Reference: IsaacGymEnvs Go2Terrain Environment
# Source: https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py
# 
# Key reward functions to reference for Isaac Lab implementation:
# - compute_reward_CaT() (line ~1050) - Main reward function
# - _reward_raibert_heuristic() (line ~620) - Footstep placement
# - _step_contact_targets() (line ~580) - Gait clock generation

import numpy as np
import os, time

from isaacgym import gymtorch
from isaacgym import gymapi

import torch
from typing import Tuple, Dict

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, normalize, quat_apply, quat_rotate_inverse, quat_rotate, quat_conjugate
from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgymenvs.utils.constraint_manager import ConstraintManager
from isaacgymenvs.tasks.terrain import Terrain
from texttable import Texttable
import itertools

class Go2Terrain(VecTask):
    """Environment to learn locomotion on complex terrains with the Solo-12 quadruped robot."""

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg
        self.init_done = False

        # Client to control the target velocity with a gamepad
        self.useJoystick = self.cfg["env"]["enableJoystick"] and self.cfg["test"]
        if self.useJoystick:
            from Joystick import Joystick

            self.joystick = Joystick()
            self.joystick.update_v_ref(0, 0)

        # Scales of observations
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.torques_scale = self.cfg["env"]["learn"]["torquesScale"]
        self.foot_forces_scale = self.cfg["env"]["learn"]["footForcesScale"]
        self.height_meas_scale = self.cfg["env"]["learn"]["heightMeasurementScale"]
        self.imu_scale = self.cfg["env"]["learn"]["imuAccelerationScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        # Scales of rewards
        self.rew_scales = {}

        # Scales for velocity tracking
        self.rew_scales["termination"] = self.cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"] 
        self.rew_scales["lin_vel_z"] = self.cfg["env"]["learn"]["linearVelocityZRewardScale"] 
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"] 
        self.rew_scales["ang_vel_xy"] = self.cfg["env"]["learn"]["angularVelocityXYRewardScale"] 
        self.rew_scales["orient"] = self.cfg["env"]["learn"]["orientationRewardScale"] 
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self.cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self.cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["air_time"] = self.cfg["env"]["learn"]["feetAirTimeRewardScale"]
        self.rew_scales["collision"] = self.cfg["env"]["learn"]["kneeCollisionRewardScale"]
        self.rew_scales["stumble"] = self.cfg["env"]["learn"]["feetStumbleRewardScale"]
        self.rew_scales["action_rate"] = self.cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["dof_pos"] = self.cfg["env"]["learn"]["dofPosRewardScale"]
        self.rew_scales["dof_vel_limit"] = self.cfg["env"]["learn"]["dofVelLimitRewardScale"]
        self.rew_scales["hip"] = self.cfg["env"]["learn"]["hipRewardScale"]
        self.rew_scales["foot2contact"] = self.cfg["env"]["learn"]["footTwoContactRewardScale"]
        self.rew_scales["raibertHeuristic"] = self.cfg["env"]["learn"]["raibertHeuristic"]
        self.rew_scales["standStill"] = self.cfg["env"]["learn"]["standStill"]
        self.rew_scales["dof_vel"] = self.cfg["env"]["learn"]["dof_vel"]
        self.rew_scales["feetClearance"] = self.cfg["env"]["learn"]["feetClearance"]
        self.rew_scales["trackingContactsShapedForce"] = self.cfg["env"]["learn"]["trackingContactsShapedForce"]
        self.rew_scales["trackingContactsShapedVel"] = self.cfg["env"]["learn"]["trackingContactsShapedVel"]
        self.lin_vel_delta = self.cfg["env"]["learn"]["linearVelocityXYRewardDelta"]
        self.ang_vel_delta = self.cfg["env"]["learn"]["angularVelocityZRewardDelta"]
        self.air_time_target = self.cfg["env"]["learn"]["feetAirTimeRewardTarget"]

        # ... (initialization continues)

    # ============================================================================
    # KEY FUNCTION: Gait clock generation
    # ============================================================================
    def _step_contact_targets(self):
        frequencies = 3. # torch.tensor([3.], device=self.device).unsqueeze(0)
        phases = 0.5 # torch.tensor([0.5], device=self.device).unsqueeze(0)
        offsets = 0. # torch.tensor([0.], device=self.device).unsqueeze(0)
        bounds = 0. # torch.tensor([0.], device=self.device).unsqueeze(0)
        durations = 0.5 * torch.ones((self.num_envs,), dtype=torch.float32, device=self.device)
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
            stance_idxs = torch.remainder(idxs, 1) < durations
            swing_idxs = torch.remainder(idxs, 1) > durations

            idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
            idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                        0.5 / (1 - durations[swing_idxs]))

        # if self.cfg.commands.durations_warp_clock_inputs:

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])

        # von mises distribution
        kappa = 0.07 # self.cfg.rewards.kappa_gait_probs
        smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

        smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
        smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
        smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
                1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                                    smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                            1 - smoothing_cdf_start(
                                        torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))

        self.desired_contact_states[:, 0] = smoothing_multiplier_FL
        self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        self.desired_contact_states[:, 2] = smoothing_multiplier_RL
        self.desired_contact_states[:, 3] = smoothing_multiplier_RR

    # ============================================================================
    # KEY FUNCTION: Raibert heuristic reward
    # ============================================================================
    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            #footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footsteps_translated[:, i, :])
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([3.0], device=self.device)
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    # ============================================================================
    # KEY FUNCTION: Main reward computation (CaT = Constraints as Terminations)
    # ============================================================================
    def compute_reward_CaT(self):
        """Compute a limited set of rewards for constraints as terminations."""

        # Velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / self.lin_vel_delta) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / self.ang_vel_delta) * self.rew_scales["ang_vel_z"]

        # Torque regularization
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # Action rate regularization
        rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]
        rew_action_rate += torch.sum(torch.square(self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]

        self.reward_actions = self.actions.clone()
        self.reward_last_actions = self.last_actions[:, :, 0].clone()
        self.reward_rew_action_rate = rew_action_rate.clone()

        # Orientation penalty (projected gravity XY)
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]
        
        # Vertical velocity penalty
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        
        # Joint velocity penalty
        rew_dof_vel = torch.sum(torch.square(self.dof_vel), dim=1) * self.rew_scales["dof_vel"]
        
        # Roll/pitch angular velocity penalty
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # Feet air time reward
        rew_airTime = torch.sum((self.feet_swing_time - 0.25) * self.contacts_touchdown, dim=1) * self.rew_scales["air_time"]

        # Penalty for having more or less than 2 feet in contact
        rew_foot2contact = - torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 2) / 2 * self.rew_scales["foot2contact"]

        # Raibert heuristic reward
        rew_raibert_heuristic = self._reward_raibert_heuristic() * self.rew_scales["raibertHeuristic"]
        
        # Stand still penalty
        rew_stand_still = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) \
            * (torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone) \
            * (torch.abs(self.commands[:, 2]) < 0.2) * self.rew_scales["standStill"]

        # ============================================================================
        # FEET CLEARANCE REWARD - Key implementation to reference
        # ============================================================================
        # phases: 0-0.5 = stance, 0.5-1.0 = swing
        # foot_indices is clipped to 0-1 range where >0.5 means swing phase
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, :, 2]).view(self.num_envs, -1)  # Z coordinate
        target_height = 0.08 * phases + 0.02  # target clearance + foot radius offset
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1) * self.rew_scales["feetClearance"]

        # ============================================================================
        # TRACKING CONTACTS SHAPED FORCE - Key implementation to reference
        # ============================================================================
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        rew_tracking_contacts_shaped_force = 0.
        for i in range(4):
            # Penalize contact force when foot should be in swing (desired_contact=0)
            rew_tracking_contacts_shaped_force += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force * self.rew_scales["trackingContactsShapedForce"] / 4

        # Total reward, with clipping if < 0
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_action_rate + rew_airTime + rew_foot2contact + rew_raibert_heuristic + rew_stand_still + rew_orient + rew_lin_vel_z + rew_dof_vel + rew_ang_vel_xy + rew_feet_clearance + rew_tracking_contacts_shaped_force

        if self.useConstraints == "cat":
            self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)
        else:
            self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # Saving the cumulative sum of rewards over the episodes
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["air_time"] += rew_airTime
        self.episode_sums["foot2contact"] += rew_foot2contact
        self.episode_sums["raibertHeuristic"] += rew_raibert_heuristic
        self.episode_sums["standStill"] += rew_stand_still
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["dof_vel"] += rew_dof_vel
        self.episode_sums["feet_clearance"] += rew_feet_clearance
        self.episode_sums["tracking_contacts_shaped_force"] += rew_tracking_contacts_shaped_force
        self.cat_discounted_cum_reward += self.cat_cum_discount_factor * self.rew_buf


# ============================================================================
# JIT HELPER FUNCTIONS
# ============================================================================
@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


