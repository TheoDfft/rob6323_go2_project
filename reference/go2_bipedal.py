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

class Go2Bipedal(VecTask):
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
        self.foot_velocity_scale = self.cfg["env"]["learn"]["footVelocityScale"]
        self.foot_ang_velocity_scale = self.cfg["env"]["learn"]["footAngVelocityScale"]
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
        self.rew_scales["torque_limits"] = self.cfg["env"]["learn"]["torqueLimitsRewardScale"]
        self.rew_scales["foot_shift"] = self.cfg["env"]["learn"]["footShift"]
        self.lin_vel_delta = self.cfg["env"]["learn"]["linearVelocityXYRewardDelta"]
        self.ang_vel_delta = self.cfg["env"]["learn"]["angularVelocityZRewardDelta"]
        self.air_time_target = self.cfg["env"]["learn"]["feetAirTimeRewardTarget"]
        self.soft_torque_limit = self.cfg["env"]["learn"]["softTorqueLimit"]

        self.numRewards = -1

        self.diffReward = self.cfg["env"].get("diffReward", False)
        if self.diffReward:
            print("Using differential reward only.")

        # TODO: Remove?
        self.rew_mult = self.cfg["env"]["learn"]["rewMult"]
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.rew_mult

        # Scales of constraints (CaT)
        self.useConstraints = self.cfg["env"]["learn"]["enableConstraints"]
        if self.useConstraints == "cat":
            # Use constraints as terminations
            self.constraints = {}
            self.constraints["survival_bonus"] = self.cfg["env"]["learn"]["constraints_CaT"]["survivalBonus"]
            self.constraints["air_time"] = self.cfg["env"]["learn"]["constraints_CaT"]["feetAirTimeConstraint"]
            self.constraints["max_air_time"] = self.cfg["env"]["learn"]["constraints_CaT"]["feetMaxAirTimeConstraint"]
            self.constraints["soft_p"] = self.cfg["env"]["learn"]["constraints_CaT"]["softPConstraint"]
            self.constraints["useSoftPCurriculum"] = self.cfg["env"]["learn"]["constraints_CaT"]["useSoftPCurriculum"]
            self.constraints["softPCurriculumMaxEpochs"] = int(self.cfg["env"]["learn"]["constraints_CaT"]["softPCurriculumMaxEpochs"])
            self.constraints["curriculum"] = 0.0
            self.constraints["tracking"] = self.cfg["env"]["learn"]["constraints_CaT"]["trackingConstraint"]
            self.cstr_manager = ConstraintManager(tau=self.cfg["env"]["learn"]["constraints_CaT"]["tauConstraint"], 
                                                  min_p=self.cfg["env"]["learn"]["constraints_CaT"]["minPConstraint"])
            self.numConstraints = -1
        else:
            self.numConstraints = 0

        # Values of constraints limits
        self.limits = self.cfg["env"]["learn"]["limits"]

        # Scales for velocity tracking commands
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        if self.cfg["env"]["onlyForwards"]:
            # Just going forwards at maximum velocity
            self.command_x_range[0] = self.command_x_range[1]
            for i in range(2):
                self.command_y_range[i] = 0.
                self.command_yaw_range[i] = 0.

        # Initial state of the robot base
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # Other miscellaneous quantities
        self.capturing_video = virtual_screen_capture
        self.height_samples = None
        self.custom_origins = False
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.debug_plots = self.cfg["env"]["enableDebugPlots"]
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self.cfg["sim"]["dt"]
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s/ self.dt + 0.5)
        self.randomize_friction = self.cfg["env"]["learn"]["randomizeFriction"]
        self.randomize_motor_friction = self.cfg["env"]["learn"]["randomizeMotorFriction"]
        self.push_enable = self.cfg["env"]["learn"]["pushRobots"]

        #self.push_interval = int(self.cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.pushInterval_s = self.cfg["env"]["learn"]["pushInterval_s"]

        self.allow_knee_contacts = self.cfg["env"]["learn"]["allowKneeContacts"]
        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]
        self.curriculum = self.cfg["env"]["terrain"]["curriculum"]
        self.base_height_target = self.cfg["env"]["learn"]["base_height_target"]
        self.phases_freq = self.cfg["env"]["learn"]["phases_freq"]
        self.gait_period = self.cfg["env"]["learn"]["gait_period"]
        self.vel_deadzone = self.cfg["env"]["learn"]["vel_deadzone"]
        self.flat_terrain_threshold = self.cfg["env"]["learn"]["flatTerrainThreshold"]

        # Compute the dimension of the observations
        self.num_height_points = len(self.cfg["env"]["learn"]["measured_points_x"]) * len(self.cfg["env"]["learn"]["measured_points_y"])
        self.prepare_dim_obs_functions()
        self.sampleObsSize = self.get_dim_observations()
        self.numHistorySamples = self.cfg["env"]["numHistorySamples"]
        self.numHistoryStep = self.cfg["env"]["numHistoryStep"]

        # Save relevant quantities for other parts of the pipeline
        self.cfg["env"]["sampleObsSize"] = self.sampleObsSize
        self.cfg["env"]["numObservations"] = self.sampleObsSize * self.numHistorySamples
        sizeObs = self.sampleObsSize * self.numHistorySamples
        sizeObsHist = self.sampleObsSize * (1 + (self.numHistorySamples - 1) * self.numHistoryStep)
        self.num_dyn_obs = self.cfg["env"]["numObservations"]
        self.num_act_obs = self.num_dyn_obs - 15 - 3 - 12 #- 1 #- 12 #obs_changes#
        self.debuild_act_nb = 3 + 12

        # Option to scale the rewards by the time step
        # for key in self.rew_scales.keys():
        #    self.rew_scales[key] *= self.dt

        # Initialization of the parent class VecTask
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # Reapply time step value because it gets overwritten in VecTask
        self.dt = self.decimation * self.cfg["sim"]["dt"]

        # Backward compatibility in gym wrapper
        self.metadata["video.frames_per_second"] = int(np.round(1/(self.dt)))
        self.metadata["video.output_frames_per_second"] = 30

        # self.gym.simulate will be called in pre_physics_step with the decimation mechanism
        # Setting this to 0 avoid an additional call in VecTask.step
        self.control_freq_inv = 0

        # Prepare the height scan
        self.height_points = self.prepare_height_points()
        self.measured_heights = torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)

        self.use_actuator_net = self.cfg["env"]["control"]["useActuatorNet"]
        if self.use_actuator_net:
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                with torch.no_grad():
                    torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

        # Prepare observations, i.e gather functions that will compute observations
        # and observation noises
        self.prepare_observation_functions()
        self.prepare_noise_functions()

        # Initialization of the camera position if there is a graphical interface
        if self.graphics_device_id != -1:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # Get gym GPU state tensors (i.e direct memory wrapping of the simulator)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)  # for imu force sensor

        # Create some wrapper tensors for different slices
        # These tensors will be updated when the associated self.gym.refresh_xxx is called
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.post_base_forward_vec = quat_apply(self.base_quat, self.forward_vec)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self.last_contact_forces = self.contact_forces.clone()
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self.force_sensor = gymtorch.wrap_tensor(force_sensor_tensor)
        self.init_feet_positions = torch.zeros((self.num_envs, 4, 3), dtype=torch.float, device=self.device)
        self.last_joint_acc = None

        # Taking only the [0:3] components breaks the automatic update like for previous tensors
        self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]
        self.foot_velocities_ang = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 10:13]
        # TODO: Find a way to have these two keep the view like dof_pos and dof_vel to avoid having to update them manually

        # Initialize some data and tensors used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self.get_noise_scale_vec(self.cfg)
        self.commands = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale], device=self.device, requires_grad=False,)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.side = "front"
        if self.side == "front":
            self.upright_vec = to_torch([0.2, 0.0, -1.0], device=self.device).repeat((self.num_envs, 1))
            self.fidx = [0, 1]
        else:
            self.upright_vec = to_torch([0.2, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
            self.fidx = [2, 3]
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.phases = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contacts_last = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contacts_touchdown = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.feet_gait_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_swing_time = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_swing_apex = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_clearance = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_clearance_cstr = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.obs_buf = torch.zeros((self.num_envs, sizeObs), dtype=torch.float, device=self.device, requires_grad=False)
        self.hist_obs_buf = torch.zeros((self.num_envs, sizeObsHist), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_quat[:, 3] = 1.0
        self.filtered_contact_forces = torch.zeros((self.num_envs, 4, 3, 5), dtype=torch.float, device=self.device, requires_grad=False)
        self.move_up_flag = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.camera_pos = np.zeros(3)
        self.desired_contact_states = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
        # Keeping the 6 last actions and joint states
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros(self.num_envs, self.num_dof, 6, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros(self.num_envs, self.num_dof, 6, dtype=torch.float, device=self.device, requires_grad=False)

        # Prepare depth cameras
        self.use_depth = self.cfg["env"]["depth"]["use_depth"]
        if self.use_depth:
            assert self.cfg["env"]["enableCameraSensors"]
            self.depth_clip = self.cfg["env"]["depth"]["depth_clip"]
            self.depth_update_interval = self.cfg["env"]["depth"]["update_interval"]
            self.depths = torch.zeros(
                    (self.num_envs, self.cfg["env"]["depth"]["image_size"][0], self.cfg["env"]["depth"]["image_size"][1]),
                    dtype=torch.float, device=self.device, requires_grad=False
            )

        # Initialize logging tensors, only log quantities of the first environment so no need for self.num_envs
        if self.debug_plots:
            self.log_feet_positions = torch.zeros(self.max_episode_length * self.decimation, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_feet_velocities = torch.zeros(self.max_episode_length * self.decimation, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_feet_ctc_forces = torch.zeros(self.max_episode_length, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_dof_pos = torch.zeros(self.max_episode_length * self.decimation, 12, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_dof_pos_cmd = torch.zeros(self.max_episode_length * self.decimation, 12, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_dof_vel = torch.zeros(self.max_episode_length * self.decimation, 12, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_torques = torch.zeros(self.max_episode_length * self.decimation, 12, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_action_rate = torch.zeros(self.max_episode_length * self.decimation, 12, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_base_vel = torch.zeros(self.max_episode_length * self.decimation, 6, dtype=torch.float, device=self.device, requires_grad=False)
            self.log_trajectory = torch.zeros(self.max_episode_length * self.decimation, 19, dtype=torch.float, device=self.device, requires_grad=False)

        # Default joint positions to which the joint position offsets (actions) are added
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_actions):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle

        # Logging rewards over the whole episodes (cumulative sum)
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"lin_vel_xy": torch_zeros(), "ang_vel_z": torch_zeros(), "torques": torch_zeros(),
                                "action_rate": torch_zeros(), "foot_shift": torch_zeros(), "torque_limits": torch_zeros(),
                                "raibertHeuristic": torch_zeros(), "standStill": torch_zeros(), "orient": torch_zeros(),
                                "lin_vel_y": torch_zeros(), "dof_vel": torch_zeros(), "ang_vel_xy": torch_zeros(),
                                "feet_clearance": torch_zeros(), "tracking_contacts_shaped_force": torch_zeros(),
                                "lift_up": torch_zeros(), "upright": torch_zeros(), "rear_air": torch_zeros(),
                                "symmetry": torch_zeros(), "slip": torch_zeros(), "stand_air": torch_zeros(),
                                "position_protect": torch_zeros()}
        self.cat_cum_discount_factor = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.cat_discounted_cum_reward = torch_zeros()

        # Reset all environments once to prepare them for the first action
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # Initialisation is done
        self.init_done = True

    def create_sim(self):
        """Create the simulation terrain and the environments (i.e. spawn the robots)."""

        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        # Either load a huge flat ground ("plane") or a complex trimesh
        terrain_type = self.cfg["env"]["terrain"]["terrainType"] 
        if terrain_type=='plane':
            self._create_ground_plane()
        elif terrain_type=='trimesh':
            self._create_trimesh()
            self.custom_origins = True

        # Initialize the environments
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """Load a huge flat plane in the simulation."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_trimesh(self):
        """Load a complex trimesh in the simulation, created with the Terrain class."""

        # Create the terrain mesh
        self.terrain = Terrain(self.cfg["env"]["terrain"], num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]

        # Load the mesh into the simulation and save the whole terrain heightmap for easy access
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Initalize the environments by spawning one robot (the actor) for each env."""

        # Getting the asset file
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
        asset_file = self.cfg["env"]["urdfAsset"]["file"]
        action_bounds_file = self.cfg["env"]["urdfAsset"]["action_bounds_file"]
        asset_path = os.path.join(asset_root, asset_file)
        action_bounds_path = os.path.join(asset_root, action_bounds_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        self.action_min, self.action_max = torch.load(action_bounds_path)
        #self.action_min, self.action_max = self.action_min.to(self.device) * 1.5, self.action_max.to(self.device) * 1.5
        self.action_min, self.action_max = self.action_min.to(self.device) / 2., self.action_max.to(self.device) / 2.
        #self.action_min, self.action_max = self.action_min.clip(-2, 0), self.action_max.clip(0, 2)
        print("action bound min", self.action_min)
        print("action bound max", self.action_max)
        self.action_scale_prime = (self.action_max - self.action_min) / 2.0
        self.action_bias = (self.action_max + self.action_min) / 2.0

        # Default asset options with joint torque control mode
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]
        asset_options.replace_cylinder_with_capsule = True
        asset_options.flip_visual_attachments = True # False
        asset_options.fix_base_link = self.cfg["env"]["urdfAsset"]["fixBaseLink"]
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        # Loading Solo-12 asset with default properties
        self.solo_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(self.solo_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.solo_asset)

        # Prepare friction randomization
        rigid_shape_prop = self.gym.get_asset_rigid_shape_properties(self.solo_asset)
        friction_range = self.cfg["env"]["learn"]["frictionRange"]
        num_buckets = 100
        friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device=self.device)

        # Prepare default base position
        self.base_init_state = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        # Loading body, DoF, knee, shin and feet names
        body_names = self.gym.get_asset_rigid_body_names(self.solo_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.solo_asset)
        foot_name = self.cfg["env"]["urdfAsset"]["footName"]
        shin_name = self.cfg["env"]["urdfAsset"]["shinName"]
        knee_name = self.cfg["env"]["urdfAsset"]["kneeName"]
        feet_names = [s for s in body_names if foot_name in s]
        shin_names = [s for s in body_names if shin_name in s]
        knee_names = [s for s in body_names if knee_name in s]
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.grf_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.shin_indices = torch.zeros(len(shin_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.base_index = 0

        # Apply the armature value to all joints (i.e. take into account the motor inertia seen at joint level)
        dof_props = self.gym.get_asset_dof_properties(self.solo_asset)
        self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(len(dof_props)):
            self.dof_pos_limits[i, 0] = dof_props["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props["upper"][i].item()
            self.dof_vel_limits[i] = dof_props["velocity"][i].item()
            self.torque_limits[i] = dof_props["effort"][i].item()
        dof_props["armature"].fill(self.cfg["env"]["urdfAsset"]["armature"])

        # Add a pseudo-Inertia Measurement Unit sensor through the use of a force sensor on the base
        # imu_pose = gymapi.Transform()
        # self.gym.create_asset_force_sensor(self.solo_asset, self.base_index, imu_pose)

        # Gather env origins and spread the robots over the whole terrain
        # The terrain is divided into rows (levels) and columns (types), each cell being a potential spawn location
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if not self.curriculum: self.cfg["env"]["terrain"]["maxInitMapLevel"] = self.cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(0, self.cfg["env"]["terrain"]["maxInitMapLevel"]+1, (self.num_envs,), device=self.device)
        self.terrain_types = torch.randint(0, self.cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device)
        if self.custom_origins:
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            spacing = 0.
        if self.cfg["test"] and self.cfg["env"]["startAtLevel"] != -1:
            # Force all robots to spawn at the same level
            self.terrain_levels[:] = self.cfg["env"]["startAtLevel"]

        # Create the env instances and save the handles to interact with them
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.solo_handles = []
        self.envs = []
        self.cam_handles = []

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)
        # Base mass randomization
        self.randomize_base_mass = self.cfg["env"]["learn"]["randomizeBaseMass"]
        min_payload, max_payload = self.cfg["env"]["learn"]["addedMassRange"]
        self.payloads = torch.rand(self.num_envs, dtype=torch.float, device=self.device, 
            requires_grad=False) * (max_payload - min_payload) + min_payload
        self.randomize_foot_mass = self.cfg["env"]["learn"]["randomizeFootMass"]
        self.randomize_hip_mass = self.cfg["env"]["learn"]["randomizeHipMass"]
        self.randomize_thigh_mass = self.cfg["env"]["learn"]["randomizeThighMass"]
        self.randomize_calf_mass = self.cfg["env"]["learn"]["randomizeCalfMass"]

        # Restitution randomization
        self.randomize_restitution = self.cfg["env"]["learn"]["randomizeRestitution"]
        min_restitution, max_restitution = self.cfg["env"]["learn"]["restitutionRange"]
        self.restitutions = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * \
            (max_restitution - min_restitution) + min_restitution

        # Com displacement randomization
        self.randomize_com_displacement = self.cfg["env"]["learn"]["randomizeComDisplacement"]
        """min_com_displacement, max_com_displacement = self.cfg["env"]["learn"]["comDisplacementRange"] #/!\#
        self.com_displacements = torch.rand(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False) * \
            (max_com_displacement - min_com_displacement) + min_com_displacement"""

        self.ratio_delay_start = self.cfg["env"]["control"].get("ratio_delay_start", 0.0)
        self.ratio_delay_end = self.cfg["env"]["control"].get("ratio_delay_end", 0.0)
        self.ratio_delay_min_steps = self.cfg["env"]["control"]["ratio_delay_min_steps"]
        self.ratio_delay_max_steps = self.cfg["env"]["control"]["ratio_delay_max_steps"] # 16000
        self.decimation_range = self.cfg["env"]["control"]["decimation_range"]

        self.lag_timesteps = self.cfg["env"]["learn"]["lag_timesteps"]
        self.swing_lag_timesteps = self.cfg["env"]["learn"]["swing_lag_timesteps"]
        self.stance_lag_timesteps = self.cfg["env"]["learn"]["stance_lag_timesteps"]
        self.lag_min_steps = self.cfg["env"]["learn"]["lag_min_steps"]

        self.lag_buffer = torch.zeros((self.num_envs, self.num_dof, self.lag_timesteps + 1), dtype=torch.float, device=self.device)
        self.lag_steps = torch.zeros((self.num_envs, self.num_dof), dtype=int, device=self.device)

        self.lag_steps[:] = torch.from_numpy(np.random.randint(
            low=self.stance_lag_timesteps[0],
            high=self.stance_lag_timesteps[1] + 1,
            size=self.lag_steps.shape
        )).to(self.device) #/!\#

        self.valid_history_length = torch.ones((self.num_envs, self.num_dof), dtype=int, device=self.device) * (self.lag_timesteps + 1)

        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_mu_v = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_Fs = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self._randomize_dof_props(torch.arange(self.num_envs))
        self.mass = torch.zeros(self.num_bodies, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            # Create one environment
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            if self.custom_origins:
                self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
                pos = self.env_origins[i].clone()
                pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
                start_pose.p = gymapi.Vec3(*pos)

            # Create one actor (robot) in that environment
            if self.randomize_friction:
                for s in range(len(rigid_shape_prop)):
                    rigid_shape_prop[s].friction = friction_buckets[i % num_buckets]

            if self.randomize_restitution:
                for s in range(len(rigid_shape_prop)):
                    rigid_shape_prop[s].restitution = self.restitutions[i]

            self.gym.set_asset_rigid_shape_properties(self.solo_asset, rigid_shape_prop)
            solo_handle = self.gym.create_actor(env_handle, self.solo_asset, start_pose, "solo", i, 0, 0)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, solo_handle)

            for j, p in enumerate(body_props):
                self.mass[j] = p.mass

            if self.randomize_base_mass:
                default_body_mass = body_props[0].mass
                body_props[0].mass = default_body_mass + self.payloads[i]

            if self.randomize_com_displacement:
                rng = [[-0.01, 0.0, -0.01], [0.01, 0.0, 0.01]] # [[-0.01, 0.0, -0.01], [0.01, 0.0, 0.01]] #/!\#
                com_displacement = np.array([np.random.uniform(rng[0][0], rng[1][0]), np.random.uniform(rng[0][1], rng[1][1]), np.random.uniform(rng[0][2], rng[1][2])])
                #body_props[0].com = gymapi.Vec3(self.com_displacements[i, 0], self.com_displacements[i, 1], \
                #    self.com_displacements[i, 2]) #/!\#also looks strange
                body_props[0].com += gymapi.Vec3(
                    com_displacement[0], com_displacement[1], com_displacement[2]
                )

            # TODO: seems a critical factor
            hip_mass_offset = np.random.uniform(0.0, 0.1)
            thigh_mass_offset = np.random.uniform(max(-0.05, -0.5 * body_props[2].mass), 0.05)
            calf_mass_offset = np.random.uniform(max(-0.05, -0.5 * body_props[3].mass), 0.05)
            foot_mass_offset = np.random.uniform(max(-0.01, -0.5 * body_props[4].mass), 0.01)
            for j in range(1, len(body_props)):
                if self.randomize_foot_mass and j % 4 == 0:
                    # otherwise, there are strange collisions on calf
                    body_props[j].mass += foot_mass_offset
                elif self.randomize_hip_mass and j % 4 == 1:
                    body_props[j].mass += hip_mass_offset
                elif self.randomize_thigh_mass and j % 4 == 2:
                    body_props[j].mass += thigh_mass_offset
                elif self.randomize_calf_mass and j % 4 == 3:
                    body_props[j].mass += calf_mass_offset

            self.gym.set_actor_rigid_body_properties(env_handle, solo_handle, body_props, recomputeInertia=True)

            self.gym.set_actor_dof_properties(env_handle, solo_handle, dof_props)
            self.envs.append(env_handle)
            self.solo_handles.append(solo_handle)
            self.attach_camera(i, env_handle, solo_handle)

        # Gather base, knee, shin and feet indices based on their name
        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], self.cfg["env"]["urdfAsset"]["baseName"])
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], knee_names[i])
        ##if not self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]:
        # If collapseFixedJoints is True, then the feet are collapsed into the shin and "shin + feet" become feet
        for i in range(len(shin_names)):
            self.shin_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], shin_names[i])
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], feet_names[i])

        print("knee indices", self.knee_indices)
        print("shin indices", self.shin_indices)
        print("feet indices", self.feet_indices)

        # If feet are collapsed, use shin for ground reaction forces, otherwise use the feet
        if False: # self.cfg["env"]["urdfAsset"]["collapseFixedJoints"]:
            for i in range(len(shin_names)):
                self.grf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], shin_names[i])
        else:
            for i in range(len(feet_names)):
                self.grf_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.solo_handles[0], feet_names[i])
        # TODO: Double-check which one to use

        # Retrieve base mass from body properties
        body_props = self.gym.get_actor_rigid_body_properties(self.envs[0], self.solo_handles[0])
        self.base_mass = body_props[self.base_index].mass

        pretrain_termination_contact_names = []
        for name in self.cfg["env"]["urdfAsset"]["pretrain_terminate_after_contacts_on"]:
            pretrain_termination_contact_names.extend([s for s in body_names if name in s])
        self.pretrain_termination_contact_indices = torch.zeros(len(pretrain_termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(pretrain_termination_contact_names)):
            self.pretrain_termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.solo_handles[0],
                                                                                        pretrain_termination_contact_names[i])
        self.pretrained = True #False #/!\#

        termination_contact_names = []
        for name in self.cfg["env"]["urdfAsset"]["terminate_after_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.solo_handles[0],
                                                                                        termination_contact_names[i])
            print(termination_contact_names[i], self.termination_contact_indices[i])

        penalized_contact_names = []
        for name in self.cfg["env"]["urdfAsset"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.solo_handles[0],
                                                                                        penalized_contact_names[i])
            print(penalized_contact_names[i], self.penalized_contact_indices[i])

        allow_contact_names = []
        for name in self.cfg["env"]["urdfAsset"]["allow_initial_contacts_on"]:
            allow_contact_names.extend([s for s in body_names if name in s])
        self.allow_initial_contact_indices = torch.zeros(len(allow_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        print("######")
        for i in range(len(allow_contact_names)):
            self.allow_initial_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.solo_handles[0],
                                                                                        allow_contact_names[i])
            print(allow_contact_names[i], self.allow_initial_contact_indices[i])

        self.initial_termination_contact_indices = [x.item() for x in self.termination_contact_indices if x not in self.allow_initial_contact_indices]
        print(self.termination_contact_indices, self.allow_initial_contact_indices, self.initial_termination_contact_indices)

        allow_subsequent_contact_names = []
        for name in self.cfg["env"]["urdfAsset"]["allow_subsequent_contacts_on"]:
            allow_subsequent_contact_names.extend([s for s in body_names if name in s])
        self.allow_subsequent_contact_indices = torch.zeros(len(allow_subsequent_contact_names), dtype=torch.long,
                                                       device=self.device, requires_grad=False)
        for i in range(len(allow_subsequent_contact_names)):
            self.allow_subsequent_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0],
                                                                                        self.solo_handles[0],
                                                                                        allow_subsequent_contact_names[i])
        self.subsequent_termination_contact_indices = [x.item() for x in self.termination_contact_indices if x not in self.allow_subsequent_contact_indices]

        self.dyn_obs_buf = torch.zeros(
            (self.num_envs, self.num_dyn_obs), device=self.device, dtype=torch.float)

    def _randomize_dof_props(self, env_ids):
        # Motor strength randomization
        self.randomize_motor_strength = self.cfg["env"]["learn"]["randomizeMotorStrength"]
        min_strength, max_strength = self.cfg["env"]["learn"]["motorStrengthRange"]
        if self.randomize_motor_strength:
            self.motor_strengths[env_ids, :] = torch.rand(len(env_ids), dtype=torch.float, device=self.device, requires_grad=False).unsqueeze(1) * \
                (max_strength - min_strength) + min_strength

        # Motor offsets randomization
        self.randomize_motor_offset = self.cfg["env"]["learn"]["randomizeMotorOffset"]
        min_offset, max_offset = self.cfg["env"]["learn"]["motorOffsetRange"]
        if self.randomize_motor_offset:
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) * \
                (max_offset - min_offset) + min_offset

        min_mu_v, max_mu_v = self.cfg["env"]["learn"]["mu_vRange"]
        min_Fs, max_Fs = self.cfg["env"]["learn"]["FsRange"]
        if self.randomize_motor_friction:
            self.motor_mu_v[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) * \
                (max_mu_v - min_mu_v) + min_mu_v
            self.motor_Fs[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) * \
                (max_Fs - min_Fs) + min_Fs

    def _step_contact_targets(self):
        frequencies = 2.5 #/!\# # torch.tensor([3.], device=self.device).unsqueeze(0)
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


    def _reward_raibert_heuristic(self):
        cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            #footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footsteps_translated[:, i, :])
            footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
                                                              cur_footsteps_translated[:, i, :])

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.125 # 0.25
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        desired_stance_length = 0.225 # 0.45
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([2.0], device=self.device)
        x_vel_des = self.commands[:, 0:1]
        yaw_vel_des = self.commands[:, 2:3]
        y_vel_des = yaw_vel_des * desired_stance_length / 2
        desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
        desired_ys_offset[:, 2:4] *= -1
        desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

        desired_ys_nom = desired_ys_nom + desired_ys_offset
        desired_xs_nom = desired_xs_nom + desired_xs_offset

        desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

        err_raibert_heuristic = torch.abs(desired_footsteps_body_frame[:, self.fidx] - footsteps_in_body_frame[:, self.fidx, 0:2]) #/!\#

        reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))

        return reward

    ####################
    # Observations
    ####################

    def prepare_dim_obs_functions(self):
        """Prepares a list of functions to compute the dimension of the obs vector
        Looks for self.dim_obs_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are names of all True
        observations in observe dict in the cfg.
        """

        # Prepare list of functions
        self.dim_obs_functions = []
        for name in self.cfg["env"]["learn"]["observe"].keys():
            if self.cfg["env"]["learn"]["observe"][name]:
                name = "dim_obs_" + name
                self.dim_obs_functions.append(getattr(self, name))

    def prepare_observation_functions(self):
        """Prepares a list of observation functions, which will be called to compute the total obs vector
        Looks for self.observe_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are names of all True
        observations in observe dict in the cfg.
        """

        # Prepare list of functions
        self.observation_functions = []
        self.observation_names = []
        for name in self.cfg["env"]["learn"]["observe"].keys():
            if self.cfg["env"]["learn"]["observe"][name]:
                self.observation_names.append(name)
                name = "observe_" + name
                self.observation_functions.append(getattr(self, name))

    def prepare_noise_functions(self):
        """Prepares a list of noise functions, which will be called to noise the observations
        Looks for self.noise_<OBSERVATION_NAME>, where <OBSERVATION_NAME> are names of all True
        observations in observe dict in the cfg.
        """

        # Prepare list of functions
        self.noise_functions = []
        for name in self.cfg["env"]["learn"]["observe"].keys():
            if self.cfg["env"]["learn"]["observe"][name]:
                name = "noise_" + name
                self.noise_functions.append(getattr(self, name))

    def get_dim_observations(self):
        """Return the size of the observation vector."""

        num = 0
        for dim_obs_function in self.dim_obs_functions:
            num += dim_obs_function()
        return num

    def get_noise_scale_vec(self, cfg):
        """Return the scale of the noise for each component of the observation."""

        self.add_noise = self.cfg["env"]["learn"]["addNoise"]

        # Initialize empty noise vec then stack noises
        noise_vec = torch.zeros(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        for noise_function in self.noise_functions:
            noise_vec = torch.cat(
                (
                    noise_vec,
                    noise_function(),
                ),
                dim=-1,
            )

        noise_vec *= self.cfg["env"]["learn"]["noiseLevel"]

        return noise_vec

    def compute_true_next_observations(self):
        # Update height scan (environment could have been reset between now and previous update before the rewards)
        self.measured_heights = self.get_heights()

        # Initialize empty obs buffer then stack observations
        obs_meas = torch.zeros(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        for observation_function in self.observation_functions:
            obs_meas = torch.cat(
                (
                    obs_meas,
                    observation_function(),
                ),
                dim=-1,
            )

        # Add noise to observation sample
        #if self.add_noise: #/!\#
        #    obs_meas += (2 * torch.rand_like(obs_meas) - 1) * self.noise_scale_vec #/!\#

        self.extras["obs_before_reset"] = obs_meas.clone()

    def compute_observations(self):
        """Compute observations, apply observation noises, handle observation history and fill observation buffer accordingly."""

        # Update height scan (environment could have been reset between now and previous update before the rewards)
        self.measured_heights = self.get_heights()

        # Initialize empty obs buffer then stack observations
        obs_meas = torch.zeros(
            0, dtype=torch.float, device=self.device, requires_grad=False
        )
        for observation_function in self.observation_functions:
            obs_meas = torch.cat(
                (
                    obs_meas,
                    observation_function(),
                ),
                dim=-1,
            )

        # Add noise to observation sample
        #if self.add_noise: #/!\#
        #    obs_meas += (2 * torch.rand_like(obs_meas) - 1) * self.noise_scale_vec #/!\#

        # Refresh history of observation for envs that have just been reset
        resetted = (self.progress_buf == 1)
        if torch.any(resetted):
            # No need to refresh last one (will be discarded just after)
            for i in range(0, (self.numHistorySamples - 1) * self.numHistoryStep):
                self.hist_obs_buf[resetted, i * self.sampleObsSize : (i+1) * self.sampleObsSize] = obs_meas[resetted, :]

        # Include new obs sample at the start of the history and discard oldest one
        self.hist_obs_buf = torch.cat((obs_meas, self.hist_obs_buf[:, :-self.sampleObsSize]), dim = -1)

        # Fill observation buffer with numHistorySamples samples, selected every numHistoryStep sample
        # Like if numHistorySamples = 3 and numHistoryStep = 2
        # then history contains [t-1, t-2, t-3, t-4, t-5] and we select [t-1, t-3, t-5]
        for i in range(self.numHistorySamples):
            j = i * self.numHistoryStep * self.sampleObsSize
            self.obs_buf[:, i * self.sampleObsSize : (i+1) * self.sampleObsSize] = self.hist_obs_buf[:, j:(j + self.sampleObsSize)]

    # ------------ dim obs functions ----------------
    def dim_obs_base_lin_vel(self):
        """Dimension of base linear velocity observations."""
        return 3

    def dim_obs_base_ang_vel(self):
        """Dimension of base linear angular observations."""
        return 3

    def dim_obs_commands(self):
        """Dimension of base commands observations."""
        return 3

    def dim_obs_misc(self):
        """Dimension of miscellaneous observations."""
        return 39 # 42 # 123

    def dim_obs_heights(self):
        """Dimension of height scan observations."""
        return self.num_height_points

    def dim_obs_phases(self):
        """Dimension of phases observations."""
        return 8

    def dim_obs_imu(self):
        """Dimension of IMU observations."""
        return 3

    def dim_obs_clock_inputs(self):
        """Dimension of clock_inputs observations."""
        return 4

    def dim_obs_projected_forward(self):
        """Dimension of clock_inputs observations."""
        return 3

    def dim_obs_friction(self):
        """Dimension of clock_inputs observations."""
        return 24

    def dim_obs_restitution(self):
        return 1

    def dim_obs_com_pos(self):
        return 3

    def dim_obs_feet_normal_force(self):
        return 4

    def dim_obs_foot_velocities(self):
        return 8

    def dim_obs_foot_ang_velocities(self):
        return 4

    def dim_obs_foot_positions(self):
        return 12 # 8

    def dim_obs_torques(self):
        return 12

    def dim_obs_feet_swing_time(self):
        return 4

    def dim_obs_pen_vec(self):
        return 3

    def dim_obs_foot_forces(self):
        return 4

    def dim_obs_base_quat(self):
        return 4


    def dim_obs_collision_forces(self):
        return self.penalized_contact_indices.shape[0]

    def dim_obs_feet_heights(self):
        return 4

    def dim_obs_forward(self):
        return 3

    def dim_obs_upright_vec(self):
        return 3

    def dim_obs_base_z(self):
        return 1

    # ------------ obs functions ----------------
    def observe_base_lin_vel(self):
        """Observe base linear velocity."""
        #return self.base_lin_vel * self.lin_vel_scale
        return quat_apply_yaw_inverse(self.base_quat, self.root_states[:, 7:10]) * self.lin_vel_scale

    def observe_base_ang_vel(self):
        """Observe base angular velocity."""
        return self.base_ang_vel * self.ang_vel_scale

    def observe_commands(self):
        """Observe base commands."""
        return self.commands[:, :3] * self.commands_scale

    def observe_misc(self):
        """Observe misc quantities.
        Use discrete joint velocities to avoid unreliable simulation reports.
        See https://forums.developer.nvidia.com/t/dof-velocity-offset-at-rest/205799/11
        """
        return torch.cat((self.projected_gravity,  # 3 # projected gravity, an image of the orientation (t)
                          self.dof_pos * self.dof_pos_scale, # 12 # joint positions (t)
                          ((self.dof_pos - self.last_dof_pos[:, :, 0]) / self.dt) * self.dof_vel_scale, #self.dof_vel * self.dof_vel_scale, # 12 # ((self.dof_pos - self.last_dof_pos[:, :, 0]) / self.dt) * self.dof_vel_scale, # joint velocities (t)
                          self.actions, # 12 # joint position targets (t - 1)
                         ), dim=-1)

    def observe_heights(self):
        """Observe height of surrounding terrain."""
        return torch.clip(self.root_states[:, 2].unsqueeze(1) - self.base_height_target - self.measured_heights, -1, 1.) * self.height_meas_scale

    def observe_phases(self):
        """Observe pairs of gait phases."""
        return torch.cat((torch.cos(self.phases),
                          torch.sin(self.phases)
                         ), dim=-1)

    def observe_imu(self):
        """Observe imu acceleration through force readings on the base.
        force_sensor stores net forces, that are the sum of external forces, contact forces and internal forces.
        A body resting on the ground will have a net force of zero."""
        return self.force_sensor[:, :3] / self.base_mass * self.imu_scale

    def observe_clock_inputs(self):
        return self.clock_inputs

    def observe_projected_forward(self):
        return self.projected_forward_vec

    def observe_friction(self):
        return torch.cat([self.motor_Fs, self.motor_mu_v], dim = -1)

    def observe_restitution(self):
        return normalize_range(self.restitutions.unsqueeze(dim=-1), self.cfg["env"]["learn"]["restitutionRange"])

    def observe_com_pos(self):
        return self.com_pos

    def observe_feet_normal_force(self):
        return self.contact_forces[:, self.feet_indices, 2] * self.foot_forces_scale

    def observe_foot_positions(self):
        """cur_footsteps_translated = self.foot_positions - self.base_pos.unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footsteps_translated[:, i, :])
            #footsteps_in_body_frame[:, i, :] = quat_apply_yaw(quat_conjugate(self.base_quat),
            #                                                 cur_footsteps_translated[:, i, :])[:, 0:2]
        return footsteps_in_body_frame.reshape(self.num_envs, -1)"""
        return self.foot_positions.reshape(self.num_envs, -1)


    def observe_foot_velocities(self):
        return self.foot_velocities[:, :, 0:2].reshape(self.num_envs, -1) * self.foot_velocity_scale

    def observe_foot_ang_velocities(self):
        return self.foot_velocities_ang[:, :, 2:].reshape(self.num_envs, -1) * self.foot_ang_velocity_scale

    def observe_torques(self):
        return self.torques * self.torques_scale

    def observe_feet_swing_time(self):
        return self.prev_feet_swing_time

    def observe_pen_vec(self):
        return self.pen_vec

    def observe_foot_forces(self):
        return torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * self.foot_forces_scale

    def observe_base_quat(self):
        return self.base_quat


    def observe_collision_forces(self):
        return torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1)

    def observe_feet_heights(self):
        return self.foot_positions[:, :, 2]

    def observe_forward(self):
        return quat_apply(self.base_quat, self.forward_vec)

    def observe_upright_vec(self):
        return quat_apply_yaw(self.base_quat, self.upright_vec)

    def observe_base_z(self):
        return self.base_pos[:, -1:]

    # ------------ noise obs functions ----------------
    def noise_base_lin_vel(self):
        """Noise for base linear velocity."""
        return self.cfg["env"]["learn"]["linearVelocityNoise"] * self.lin_vel_scale * torch.ones(self.dim_obs_base_lin_vel(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_base_ang_vel(self):
        """Noise for base angular velocity."""
        return self.cfg["env"]["learn"]["angularVelocityNoise"] * self.ang_vel_scale * torch.ones(self.dim_obs_base_ang_vel(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_commands(self):
        """Noise for base commands."""
        return torch.zeros(self.dim_obs_commands(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_misc(self):
        """Noise for misc quantities."""
        return torch.cat((self.cfg["env"]["learn"]["gravityNoise"] * torch.ones(3, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofPositionNoise"] * self.dof_pos_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofVelocityNoise"] * self.dof_vel_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          torch.zeros(12, dtype=torch.float, device=self.device, requires_grad=False),
                          ), dim=-1)
        return torch.cat((self.cfg["env"]["learn"]["gravityNoise"] * torch.ones(3, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofPositionNoise"] * self.dof_pos_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          torch.zeros(36, dtype=torch.float, device=self.device, requires_grad=False),
                          self.cfg["env"]["learn"]["dofVelocityNoise"] * self.dof_vel_scale * torch.ones(12, dtype=torch.float, device=self.device, requires_grad=False),
                          torch.zeros(36 + 24, dtype=torch.float, device=self.device, requires_grad=False),
                          ), dim=-1)

    def noise_heights(self):
        """Noise for height of surrounding terrain."""
        return self.cfg["env"]["learn"]["heightMeasurementNoise"] * self.height_meas_scale * torch.ones(self.dim_obs_heights(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_phases(self):
        """Noise for pairs of gait phases."""
        return torch.zeros(self.dim_obs_phases(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_imu(self):
        """Noise for IMU readings."""
        return torch.zeros(self.dim_obs_imu(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_clock_inputs(self):
        return torch.zeros(self.dim_obs_clock_inputs(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_projected_forward(self):
        return torch.zeros(self.dim_obs_projected_forward(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_friction(self):
        return torch.zeros(self.dim_obs_friction(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_restitution(self):
        return torch.zeros(self.dim_obs_restitution(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_com_pos(self):
        return torch.zeros(self.dim_obs_com_pos(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_feet_normal_force(self):
        return torch.zeros(self.dim_obs_feet_normal_force(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_foot_positions(self):
        return torch.zeros(self.dim_obs_foot_positions(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_foot_velocities(self):
        return torch.zeros(self.dim_obs_foot_velocities(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_foot_ang_velocities(self):
        return torch.zeros(self.dim_obs_foot_ang_velocities(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_torques(self):
        return torch.zeros(self.dim_obs_torques(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_feet_swing_time(self):
        return torch.zeros(self.dim_obs_feet_swing_time(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_pen_vec(self):
        return torch.zeros(self.dim_obs_pen_vec(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_foot_forces(self):
        return torch.zeros(self.dim_obs_foot_forces(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_base_quat(self):
        return torch.zeros(self.dim_obs_base_quat(), dtype=torch.float, device=self.device, requires_grad=False)


    def noise_collision_forces(self):
        return torch.zeros(self.dim_obs_collision_forces(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_feet_heights(self):
        return torch.zeros(self.dim_obs_feet_heights(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_forward(self):
        return torch.zeros(self.dim_obs_forward(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_upright_vec(self):
        return torch.zeros(self.dim_obs_upright_vec(), dtype=torch.float, device=self.device, requires_grad=False)

    def noise_base_z(self):
        return torch.zeros(self.dim_obs_base_z(), dtype=torch.float, device=self.device, requires_grad=False)

    ####################
    # Depth cameras
    ####################

    def attach_camera(self, i, env_handle, actor_handle):
        if self.cfg["env"]["depth"]["use_depth"]:
            camera_props = gymapi.CameraProperties()
            camera_props.width = self.cfg["env"]["depth"]["image_size"][1]
            camera_props.height = self.cfg["env"]["depth"]["image_size"][0]
            camera_props.enable_tensors = True
            camera_props.horizontal_fov = self.cfg["env"]["depth"]["horizontal_fov"]

            camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
            assert camera_handle != -1, "The camera failed to be created"
            self.cam_handles.append(camera_handle)

            local_transform = gymapi.Transform()

            camera_position = np.copy(self.cfg["env"]["depth"]["position"])
            camera_angle = np.random.uniform(self.cfg["env"]["depth"]["angle"][0], self.cfg["env"]["depth"]["angle"][1])

            local_transform.p = gymapi.Vec3(*camera_position)
            local_transform.r = gymapi.Quat.from_euler_zyx(0, np.radians(camera_angle), 0)
            root_handle = self.gym.get_actor_root_rigid_body_handle(env_handle, actor_handle)

            self.gym.attach_camera_to_body(camera_handle, env_handle, root_handle, local_transform, gymapi.FOLLOW_TRANSFORM)

    def update_depth_buffer(self):
        if not self.use_depth:
            return

        if self.common_step_counter % self.depth_update_interval != 0:
            return

        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.fetch_results(self.sim, True)

        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = -torch.clip(depth_image, min=-self.depth_clip, max=0.0)  / self.depth_clip
            self.depths[i] = depth_image

        self.gym.end_access_image_tensors(self.sim)
        self.extras["depth"] = self.depths

    ####################
    # Rewards
    ####################

    def compute_reward_CaT(self):
        """Compute a limited set of rewards for constraints as terminations."""

        # Velocity tracking reward
        #lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        #ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        #rew_lin_vel_xy = torch.exp(-lin_vel_error / self.lin_vel_delta) * self.rew_scales["lin_vel_xy"]
        #rew_ang_vel_z = torch.exp(-ang_vel_error / self.ang_vel_delta) * self.rew_scales["ang_vel_z"]

        # Torque regularization
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]
        rew_torque_limits = torch.sum((torch.abs(self.torques) - self.torque_limits * self.soft_torque_limit).clip(min=0.), dim=1) * self.rew_scales["torque_limits"]
        """rew_torque = torch.sum(
            torch.clip(torch.square(self.torque_limits) - torch.square(self.torques), min=0., max=None), dim=1
        ) * self.rew_scales["torque"]"""

        # Action rate regularization
        ##rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]) +
        ##                            torch.square(self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]),
        ##                            dim=1) * (self.action_scale**2) * self.rew_scales["action_rate"]
        rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]
        rew_action_rate += torch.sum(torch.square(self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]

        self.reward_actions = self.actions.clone()
        self.reward_last_actions = self.last_actions[:, :, 0].clone()

        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_lin_vel_y = torch.square(self.base_lin_vel[:, 1]) * (-2.0)

        ##rew_dof_vel = torch.sum(torch.square(self.dof_vel), dim=1) * self.rew_scales["dof_vel"] #/!\#
        dof_vel = ((self.dof_pos - self.last_dof_pos[:, :, 0]) / self.dt) #/!\#
        rew_dof_vel = torch.sum(torch.square(dof_vel), dim=1) * self.rew_scales["dof_vel"] #/!\#

        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # Feet air time reward
        rew_airTime = torch.sum((self.feet_swing_time - 0.25) * self.contacts_touchdown, dim=1) * self.rew_scales["air_time"]

        # Penalty for having more or less than 2 feet in contact
        rew_foot2contact = - torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 2) / 2 * self.rew_scales["foot2contact"]

        rew_raibert_heuristic = self._reward_raibert_heuristic() * self.rew_scales["raibertHeuristic"]
        rew_stand_still = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) \
            * (torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone) \
            * (torch.abs(self.commands[:, 2]) < 0.2) * self.rew_scales["standStill"]

        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states
        rew_tracking_contacts_shaped_force = 0.
        #for i in range(4): #/!\#
        for i in self.fidx:
            rew_tracking_contacts_shaped_force += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force * self.rew_scales["trackingContactsShapedForce"] / 4

        #########################################################################################################
        allow_contact_steps = 30
        allow_lin_vel_xy_steps = 60
        start_com_steps = 30
        scale_factor_low = 0.33
        scale_factor_high = 0.45
        condition = self.progress_buf > allow_contact_steps
        condition_lin_vel = self.progress_buf > allow_lin_vel_xy_steps
        condition_com = self.progress_buf > start_com_steps

        actual_lin_vel = quat_apply_yaw_inverse(self.base_quat, self.root_states[:, 7:10]) #@#
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - actual_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error/self.lin_vel_delta) # self.cfg.rewards.tracking_sigma)
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9 #0.8 #/!\#
        scaling_factor = (torch.clip(
            self.root_states[:, 2], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        rew_lin_vel_xy = rew_lin_vel_xy * is_stand.float() * scaling_factor
        rew_lin_vel_xy *= self.rew_scales["lin_vel_xy"]
        rew_lin_vel_xy_ = rew_lin_vel_xy.clone()

        rew_lin_vel_xy = torch.square(actual_lin_vel[:, :2]).sum(-1)
        rew_lin_vel_xy = rew_lin_vel_xy * condition_lin_vel.float()
        rew_lin_vel_xy *= -0.4 # -0.2 #/!\#
        rew_lin_vel_xy += rew_lin_vel_xy_

        rew_ang_vel_z = torch.square(self.base_ang_vel[:, 2]) * -0.1 * condition.float() #/!\#-0.1#

        rew_ang_vel_xz = torch.sum(torch.square(self.base_ang_vel[:, [0, 2]]), dim=1)
        rew_ang_vel_xz *= -0.04

        rew_front_symmetry = torch.square(self.dof_pos[:, 0] - self.dof_pos[:, 3])
        rew_front_symmetry += torch.square(self.dof_pos[:, 1] - self.dof_pos[:, 4])
        rew_front_symmetry += torch.square(self.dof_pos[:, 2] - self.dof_pos[:, 5])
        rew_rear_symmetry = torch.square(self.dof_pos[:, 6] - self.dof_pos[:, 9])
        rew_rear_symmetry += torch.square(self.dof_pos[:, 7] - self.dof_pos[:, 10])
        rew_rear_symmetry += torch.square(self.dof_pos[:, 8] - self.dof_pos[:, 11])
        rew_symmetry = (rew_front_symmetry * (~condition).float() + rew_rear_symmetry) * -0.0

        rew_back_still = torch.sum(torch.square(self.dof_vel[:, 6:]), dim=1) * is_stand.float() * condition_lin_vel.float() * 0. # (-0.01)

        """heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        heading_error = torch.square(wrap_to_pi(self.commands[:, 2] - heading) / np.pi)
        heading_rew = torch.exp(-heading_error / self.lin_vel_delta)
        heading_rew = heading_rew * is_stand.float() * scaling_factor
        heading_rew *= self.rew_scales["ang_vel_z"]
        rew_ang_vel_z = heading_rew"""

        """ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_ang_vel_z = torch.exp(-ang_vel_error / self.ang_vel_delta)
        rew_ang_vel_z = rew_ang_vel_z * is_stand.float() * scaling_factor
        rew_ang_vel_z *= self.rew_scales["ang_vel_z"]"""
        #rew_ang_vel_z = rew_ang_vel_xz

        rew_raibert_heuristic = rew_raibert_heuristic * is_stand.float() * condition.float()
        #rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force * is_stand.float() * condition_lin_vel.float()
        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force * is_stand.float()

        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices[:, self.fidx] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_positions[:, self.fidx, 2]).view(self.num_envs, -1) # - reference_heights
        target_height = 0.08 * phases + 0.02 # offset for foot radius 2cm #/!\#0.03
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, self.fidx])
        #rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1) * self.rew_scales["feetClearance"] * condition_lin_vel.float() * is_stand.float()
        rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1) * self.rew_scales["feetClearance"] * condition.float()

        root_height = self.root_states[:, 2]
        lift_up_threshold = [0.27, 0.55]
        rew_lift_up = (root_height - lift_up_threshold[0]) / (lift_up_threshold[1] - lift_up_threshold[0])
        rew_lift_up = torch.clamp(rew_lift_up, 0., 1.) * 0.5 # 1. # 0.5

        rew_fall = (lift_up_threshold[0] - root_height).clip(min=0.) * (-1.)

        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
        rew_upright = torch.square(0.5 * cosine_dist + 0.5) * 1.0

        contact = self.contact_forces[:, self.feet_indices[self.fidx], 2] < 1.
        calf_contact = self.contact_forces[:, self.knee_indices[self.fidx], 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        rew_rear_air = (torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()) * (-0.5)

        # center of mass
        feet_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3] - self.root_states[:, :3].unsqueeze(1)
        feet_pos[:, 0] = quat_rotate_inverse(self.root_states[:, 3:7], feet_pos[:, 0])
        feet_pos[:, 1] = quat_rotate_inverse(self.root_states[:, 3:7], feet_pos[:, 1])
        feet_pos[:, 2] = quat_rotate_inverse(self.root_states[:, 3:7], feet_pos[:, 2])
        feet_pos[:, 3] = quat_rotate_inverse(self.root_states[:, 3:7], feet_pos[:, 3])
        feet_normal_force = self.contact_forces[:, self.feet_indices, 2]
        cop_pos = (1./(torch.sum(feet_normal_force, dim=-1)+1e-6)).unsqueeze(-1) * torch.sum(feet_pos * feet_normal_force.unsqueeze(-1), dim=1)

        com_cop = self.com_pos - cop_pos
        pen_vec = quat_apply(self.root_states[:, 3:7], com_cop)

        pen_len = torch.norm(pen_vec, dim=1)
        pen_ang = torch.acos(pen_vec[:,2] / (1e-6 * torch.ones(self.num_envs, device=self.device, requires_grad=False) + pen_len))
        rew_inv_pendulum = torch.pow(pen_ang, 2) * condition_com.float() * torch.all(~contact, dim=1).float() * (-100.)

        rew_base_height = torch.square(self.root_states[:, 2] - self.base_height_target) * self.rew_scales["base_height"]

        rew_collision = torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 0.1), dim=1) * (-1.0) # (-2.0) #@#

        #rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1) * self.rew_scales["action_rate"]

        rew_action_q_diff = torch.sum(torch.square(self.q_diff_buf), dim=-1) * (~condition).float() * (-0.5 * 2)

        """stand_air_condition = torch.logical_and(
            torch.logical_and(self.progress_buf > 3, self.progress_buf <= allow_contact_steps),
            torch.any(self.foot_positions[:, self.fidx, 2] > 0.06, dim=-1)
        )"""
        rew_stand_air = (self.foot_positions[:, self.fidx, 2] - 0.06).clip(min=0.).sum(-1) * (-40.0)
        rew_stand_air += (self.foot_positions[:, [2, 3], 2]).clip(max=0.06).sum(-1) * (5.0)
        rew_stand_air *= (~condition).float() #/!\#/?\#

        dof_pos_lower = (self.dof_pos_limits[:, 0] - self.dof_pos).clip(min=0.)
        dof_pos_upper = (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        rew_position_protect = (dof_pos_lower.sum(-1) + dof_pos_upper.sum(-1)) * (-100.)

        desired_foot_positions = torch.clone(self.init_feet_positions[:, self.fidx])
        desired_foot_positions[:, :, 2] = 0.02
        front_foot_shift = torch.norm(self.foot_positions[:, self.fidx] - desired_foot_positions, dim=-1).mean(dim=1)
        init_rfoot_positions = torch.clone(self.init_feet_positions[:, [2, 3]])
        rear_foot_shift = torch.norm( torch.stack([
            (init_rfoot_positions[:, :, 0] - self.foot_positions[:, [2, 3], 0]).clamp(min=0), # must go forward #/!\#
            torch.abs(init_rfoot_positions[:, :, 1] - self.foot_positions[:, [2, 3], 1])
        ], dim=-1), dim=-1).mean(dim=1)
        rew_foot_shift = (front_foot_shift + rear_foot_shift) * (~condition).float() * self.rew_scales["foot_shift"]

        # Slip
        slip_condition = self.foot_positions[:, :, 2] < 0.03
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self.foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(slip_condition.float() * (foot_velocities + foot_ang_velocities), dim=1) * (-0.04 * 10)

        # Total reward, with clipping if < 0
        self.rew_buf = rew_upright + rew_lift_up + rew_stand_air + rew_lin_vel_xy + rew_feet_clearance + rew_tracking_contacts_shaped_force + rew_torque_limits + rew_ang_vel_z + rew_action_rate + rew_dof_vel

        if self.useConstraints == "cat":
            self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)
        else:
            self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # Saving the cumulative sum of rewards over the episodes
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["foot_shift"] += rew_foot_shift
        self.episode_sums["torque_limits"] += rew_torque_limits
        self.episode_sums["raibertHeuristic"] += rew_raibert_heuristic
        self.episode_sums["standStill"] += rew_stand_still
        self.episode_sums["orient"] += rew_orient
        self.episode_sums["lin_vel_y"] += rew_lin_vel_y
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["dof_vel"] += rew_dof_vel
        self.episode_sums["feet_clearance"] += rew_feet_clearance
        self.episode_sums["tracking_contacts_shaped_force"] += rew_tracking_contacts_shaped_force
        self.episode_sums["lift_up"] += rew_lift_up
        self.episode_sums["upright"] += rew_upright
        self.episode_sums["rear_air"] += rew_rear_air
        self.episode_sums["symmetry"] += rew_symmetry
        self.episode_sums["slip"] += rew_slip
        self.episode_sums["position_protect"] += rew_position_protect
        self.episode_sums["stand_air"] += rew_stand_air
        self.cat_discounted_cum_reward += self.cat_cum_discount_factor * self.rew_buf

    def compute_reward(self):
        """Compute and stack various rewards for base velocity tracking."""

        # Velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / self.lin_vel_delta) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / self.ang_vel_delta) * self.rew_scales["ang_vel_z"]

        # Other base velocity penalties
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # Orientation penalty
        rew_orient = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) * self.rew_scales["orient"]

        # Base height penalty
        rew_base_height = torch.square(self.root_states[:, 2] - self.base_height_target) * self.rew_scales["base_height"]

        # Torque regularization
        rew_torque = torch.sum(torch.square(self.torques), dim=1) * self.rew_scales["torque"]

        # Joint acc regularization
        rew_joint_acc = torch.sum(torch.square(self.diff_dof_vel - self.last_dof_vel[:, :, 0]), dim=1) * self.rew_scales["joint_acc"]

        # Collision penalty
        knee_contact = torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.
        rew_collision = torch.sum(knee_contact, dim=1) * self.rew_scales["collision"] # sum vs any ?

        # Stumbling penalty
        stumble = (torch.norm(self.contact_forces[:, self.grf_indices, :2], dim=2) > 5.) * (torch.abs(self.contact_forces[:, self.grf_indices, 2]) < 1.)
        rew_stumble = torch.sum(stumble, dim=1) * self.rew_scales["stumble"]

        # Action rate regularization
        ##rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]) +
        ##                            torch.square(self.actions - 2 * self.last_actions[:, :, 0] + self.last_actions[:, :, 1]),
        ##                            dim=1) * (self.action_scale**2) * self.rew_scales["action_rate"]
        rew_action_rate = torch.sum(torch.square(self.actions - self.last_actions[:, :, 0]), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]

        # Deviation from initial pose regularization
        rew_dof_pos = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1) * self.rew_scales["dof_pos"]
        # rew_dof_pos *= torch.norm(self.commands[:, :3], dim=1) < self.vel_deadzone

        # Feet air time reward
        rew_airTime = torch.sum((self.feet_swing_time - 0.25) * self.contacts_touchdown, dim=1) * self.rew_scales["air_time"] # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :3], dim=1) > self.vel_deadzone # no reward for zero command

        # Penalize dof velocities over the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        rew_dof_vel_limit = torch.sum((torch.abs(self.diff_dof_vel) - 12).clip(min=0.0, max=1.0), dim=1) * self.rew_scales["dof_vel_limit"]

        # Cosmetic penalty for hip motion
        rew_hip = torch.sum(torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)* self.rew_scales["hip"]

        # Non-moving static gait
        # static = ~torch.any(torch.abs(self.commands[:, :3]) > self.vel_deadzone, dim=1)
        # rew_dof_pos *= static

        # rew_torque[static] *= 100
        # rew_action_rate[static] *= 200

        # Total reward, with clipping if < 0
        self.rew_buf = rew_lin_vel_xy + rew_ang_vel_z + rew_lin_vel_z + rew_ang_vel_xy + rew_orient + rew_base_height +\
                    rew_torque + rew_joint_acc + rew_collision + rew_action_rate + rew_dof_pos + rew_airTime + rew_hip +\
                    rew_stumble + rew_dof_vel_limit
        self.rew_buf = torch.clip(self.rew_buf, min=0., max=None)

        # Add termination reward
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        # Saving the cumulative sum of rewards over the episodes
        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["torques"] += rew_torque
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["collision"] += rew_collision
        self.episode_sums["stumble"] += rew_stumble

        """self.episode_sums["orient"] += rew_orient
        self.episode_sums["joint_acc"] += rew_joint_acc
        self.episode_sums["stumble"] += rew_stumble
        self.episode_sums["dof_pos"] += rew_dof_pos
        self.episode_sums["dof_vel_limit"] += rew_dof_vel_limit
        self.episode_sums["air_time"] += rew_airTime
        self.episode_sums["base_height"] += rew_base_height
        self.episode_sums["hip"] += rew_hip"""

    ####################
    # CaT Constraints
    ####################

    def compute_constraints_cat(self):
        """Compute various constraints for constraints as terminations. Constraints violations are asssessed then
        handed out to a constraint manager (ConstraintManager class) that will compute termination probabilities."""

        # ------------ Soft constraints ----------------

        # Torque constraint
        #cstr_torque = torch.abs(self.torques) - self.limits["torque"]

        # Joint velocity constraint
        #cstr_joint_vel = torch.abs(self.dof_vel) - self.limits["vel"]

        # urdf limits
        cstr_joint_vel = torch.abs(self.dof_vel) - self.dof_vel_limits
        cstr_torque = torch.abs(self.torques) - self.torque_limits
        cstr_dof_pos_lower = self.dof_pos_limits[:, 0] - self.dof_pos
        cstr_dof_pos_upper = self.dof_pos - self.dof_pos_limits[:, 1]

        # Joint acceleration constraint
        joint_acc = torch.abs(self.last_dof_vel[:, :, 0] - self.dof_vel) / self.dt
        cstr_joint_acc = joint_acc - self.limits["acc"]
        if self.last_joint_acc is None:
            self.last_joint_acc = torch.zeros_like(joint_acc)
        joint_jerk = torch.abs(joint_acc - self.last_joint_acc) / self.dt
        self.last_joint_acc = joint_acc.clone()
        cstr_joint_jerk = joint_jerk - self.limits["jerk"]

        # Base height constraints
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            base_height = self.root_states[:, 2]
        else:
            base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        cstr_base_height_max = base_height - self.limits['base_height_max']
        cstr_base_height_min = base_height < self.limits['base_height_min']
        cstr_base_height_min_soft = self.limits['base_height_min_soft'] - base_height

        # Action rate constraint (for command smoothness)
        cstr_action_rate = torch.abs(self.actions - self.last_actions[:, :, 0]) / self.dt - self.limits["action_rate"]

        # ------------ Hard constraints ----------------

        # Knee contact constraint
        cstr_knee_contact = torch.any(torch.norm(self.contact_forces[:, self.knee_indices, :], dim=2) > 1.0, dim=1)
        cstr_thigh_contact = torch.any(torch.norm(self.contact_forces[:, self.shin_indices, :], dim=2) > 1.0, dim=1)

        # Base contact constraint
        #cstr_base_contact = torch.norm(self.contact_forces[:, self.base_index, :], dim=1) > 1.0

        # Foot contact force constraint
        cstr_foot_contact = torch.norm(self.contact_forces[:, self.grf_indices], dim=2) - self.limits["foot_contact_force"]
        cstr_curriculum_foot_contact = torch.norm(self.contact_forces[:, self.grf_indices], dim=2) - 80
        cstr_foot_contact_vertical = torch.abs(self.contact_forces[:, self.grf_indices, 2]) - self.limits["foot_contact_vertical_force"]
        #cstr_foot_contact_rate = torch.norm(self.contact_forces[:, self.grf_indices] - self.last_contact_forces[:, self.grf_indices], dim=2) - self.limits["foot_contact_force_rate"]
        cstr_foot_contact_rate = torch.abs(torch.norm(self.contact_forces[:, self.grf_indices], dim=2) - torch.norm(self.last_contact_forces[:, self.grf_indices], dim=2)) - self.limits["foot_contact_force_rate"]
        self.last_contact_forces = self.contact_forces.clone()

        # Contacts causing early stopping
        cstr_termination_contacts = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # Stumbling constraint
        cstr_foot_stumble = torch.norm(self.contact_forces[:, self.grf_indices, :2], dim=2) - 4.0 * torch.abs(self.contact_forces[:, self.grf_indices, 2])

        # Constraint on the two front HFE joints
        cstr_HFE = torch.abs(self.dof_pos[:, [1, 4]]) - self.limits["HFE"] # Only front legs

        # Constraint to avoid having the robot upside-down
        cstr_upsidedown = self.projected_gravity[:, 2] > 0

        # ------------ Style constraints ----------------

        # Hip constraint (style constraint on HAA joint)
        cstr_HAA = torch.abs(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]) - self.limits["HAA"]
        cstr_HAA *= (torch.abs(self.commands[:, 1]) < 0.1).float().unsqueeze(1) # only constraint the hips when going straight forward

        # Base orientation constraint (style constraint on roll/pitch angles)
        cstr_base_orientation = torch.norm(self.projected_gravity[:, :2], dim=1) - self.limits["base_orientation"]

        zero_command_active = torch.logical_and(
            torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone,
            torch.abs(self.commands[:, 2]) < self.vel_deadzone
        )
        # Air time constraint (style constraint)
        ##cstr_air_time = (self.constraints["air_time"] - self.feet_swing_time) * self.contacts_touchdown * (torch.norm(self.commands[:, :3], dim=1) > self.vel_deadzone).float().unsqueeze(1)
        ##cstr_max_air_time = (self.feet_swing_time - self.constraints["max_air_time"]) * self.contacts_touchdown * (torch.norm(self.commands[:, :3], dim=1) > self.vel_deadzone).float().unsqueeze(1)
        cstr_air_time = (self.constraints["air_time"] - self.feet_swing_time) * self.contacts_touchdown * (~zero_command_active).unsqueeze(1)
        cstr_max_air_time = (self.feet_swing_time - self.constraints["max_air_time"]) * self.contacts_touchdown * (~zero_command_active).unsqueeze(1)

        # Constraint to stand still when the velocity command is 0 (style constraint)
        ##cstr_nomove = (torch.abs(self.dof_vel) - 4.0) * zero_command_active.float().unsqueeze(1)
        ##cstr_nomove = (torch.norm(self.base_lin_vel[:, :2], dim=-1) - 0.10) * zero_command_active
        ##cstr_nomove = (torch.abs(self.dof_pos - self.default_dof_pos) - 0.20) * zero_command_active.float().unsqueeze(1)
        ##cstr_nomove_vel = (torch.abs(self.dof_vel) - 4.0) * zero_command_active.float().unsqueeze(1)
        cstr_nomove = torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 4).float() * zero_command_active.float()

        # Constraint to have exactly 2 feet in contact with the ground at any time when walking (style constraint)
        #cstr_2footcontact = torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 2) * (torch.norm(self.commands[:, :3], dim=1) > 0.5).float()
        cstr_2footcontact = torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 2).float() * (~zero_command_active)
        cstr_diagfootcontact = (1.0 - torch.logical_or(
            (self.contact_forces[:, self.grf_indices[[0,3]], 2] > 1.0).all(1),
            (self.contact_forces[:, self.grf_indices[[1,2]], 2] > 1.0).all(1),
        ).float()) * (~zero_command_active)

        # Apply aesthetics constraints only on flat terrains
        cstr_HAA *= self.is_flat_terrain.unsqueeze(1)
        cstr_base_orientation *= self.is_flat_terrain
        cstr_air_time *= self.is_flat_terrain.unsqueeze(1)
        cstr_2footcontact *= self.is_flat_terrain
        cstr_nomove *= self.is_flat_terrain #.unsqueeze(1)
        ##cstr_nomove_vel *= self.is_flat_terrain.unsqueeze(1)

        # ------------ Tracking constraints ----------------

        # Velocity tracking constraints
        cstr_lin_vel = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) - self.constraints["tracking"]
        cstr_ang_vel = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]) - self.constraints["tracking"]

        # ------------ Applying constraints ----------------

        # Maximum termination probability for soft and style constraints
        soft_p = self.constraints["soft_p"]

        # Curriculum on soft_p value (optionnal)
        # Soft and style constraints are initially less enforced to let the policy explore more.
        if self.constraints["useSoftPCurriculum"]:
            ##step_cur = 1.0 / (self.cfg["horizon_length"] * self.cfg["max_epochs"])
            step_cur = 1.0 / (self.cfg["horizon_length"] * self.constraints["softPCurriculumMaxEpochs"])
            self.constraints["curriculum"] = min(self.constraints["curriculum"] + step_cur, 1.0)

            # Linearly interpolate the expected time for episode end: soft_p is the maximum
            # termination probability so it is an image of the expected time of death.
            T_start = 20
            T_end = 1 / soft_p
            soft_p = 1 / (T_start + self.constraints["curriculum"] * (T_end - T_start))

        if self.constraints["useSoftPCurriculum"]:
            self.constraints["late_curriculum"] = max((self.common_step_counter - 600 * self.cfg["horizon_length"]) / ((self.cfg["max_epochs"] - 600) * self.cfg["horizon_length"]), 0)
            soft_p_2 = self.constraints["late_curriculum"] * self.constraints["soft_p"]

        allow_contact_steps = 30 #@#
        max_dof_change = 0.75 #/!\#
        if self.pretrained:
            termination_contact_indices = self.termination_contact_indices
        else:
            termination_contact_indices = self.pretrain_termination_contact_indices

        """any_contacts = torch.logical_and(
            torch.any(torch.norm(self.contact_forces[:, termination_contact_indices, :], dim=-1) > 1., dim=1),
            torch.logical_not(torch.logical_and(
                torch.any(torch.norm(self.contact_forces[:, self.allow_initial_contact_indices, :], dim=-1) > 1., dim=1),
                self.progress_buf <= allow_contact_steps
            ))
        )""" #/!\#
        any_contacts = torch.logical_or(
            torch.any(torch.norm(self.contact_forces[:, self.allow_subsequent_contact_indices, :], dim=-1) > 1., dim=1),
            torch.logical_and(
                torch.any(torch.norm(self.contact_forces[:, termination_contact_indices, :], dim=-1) > 1., dim=1),
                torch.logical_not(torch.logical_and(
                    torch.any(torch.norm(self.contact_forces[:, self.allow_initial_contact_indices, :], dim=-1) > 1., dim=1),
                    self.progress_buf <= allow_contact_steps
                ))
            ) #/!\#
        )
        """any_contacts = torch.logical_or(
            torch.logical_and(
                self.progress_buf <= allow_contact_steps,
                torch.any(torch.norm(self.contact_forces[:, self.initial_termination_contact_indices, :], dim=-1) > 1., dim=1)
            ),
            torch.logical_and(
                self.progress_buf > allow_contact_steps,
                torch.any(torch.norm(self.contact_forces[:, self.subsequent_termination_contact_indices, :], dim=-1) > 1., dim=1)
            )
        )""" #/!\#
        # For debug purposes:
        """if any_contacts[0]:
            didx = (torch.norm(self.contact_forces[0, termination_contact_indices, :], dim=-1) > 1.).argwhere()[0]
            print(didx, self.progress_buf[0])
            exit(0)"""

        stand_air_condition = torch.logical_and(
            torch.logical_and(self.progress_buf > 3, self.progress_buf <= allow_contact_steps),
            torch.any(self.foot_positions[:, self.fidx, 2] > 0.06, dim=-1)
        )
        abrupt_change_condition = torch.logical_and(
            torch.logical_and(self.progress_buf > 3, self.progress_buf <= allow_contact_steps),
            torch.any(torch.abs(self.dof_pos - self.last_dof_pos[:, :, 0]) > max_dof_change, dim=-1)
        )
        position_protect = torch.logical_and(
            self.progress_buf > 3, torch.any(torch.logical_or(
            self.dof_pos < self.dof_pos_limits[:, 0] + 5 / 180 * np.pi,
            self.dof_pos > self.dof_pos_limits[:, 1] - 5 / 180 * np.pi
        ), dim=-1))

        allow_not_stand_steps = 60
        allow_not_stand_steps_2 = 90
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9 #/!\#
        not_stand = torch.logical_and(
            self.progress_buf > allow_not_stand_steps,
            ~is_stand
        )
        """is_stand_2 = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9
        not_stand_2 = torch.logical_and(
            self.progress_buf > allow_not_stand_steps_2,
            ~is_stand_2
        )"""

        #any_contacts = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
        #                           dim=1) #@#

        # Soft constraints
        self.cstr_manager.add("dof_pos_lower", cstr_dof_pos_lower, max_p=soft_p)
        self.cstr_manager.add("dof_pos_upper", cstr_dof_pos_upper, max_p=soft_p)
        self.cstr_manager.add("torque", cstr_torque, max_p=soft_p)
        #self.cstr_manager.add("joint_jerk", cstr_joint_jerk, max_p=soft_p)
        #self.cstr_manager.add("joint_acc", cstr_joint_acc, max_p=soft_p)
        self.cstr_manager.add("joint_vel",  cstr_joint_vel, max_p=soft_p)
        self.cstr_manager.add("base_height_max", cstr_base_height_max, max_p=soft_p)
        #self.cstr_manager.add("base_height_min", cstr_base_height_min_soft, max_p=soft_p)
        self.cstr_manager.add("action_rate", cstr_action_rate, max_p=soft_p)
        self.cstr_manager.add("foot_contact_rate", cstr_foot_contact_rate, max_p=soft_p)
        #self.cstr_manager.add("late_curriculum_foot_contact", cstr_curriculum_foot_contact, max_p=soft_p_2)

        # Hard constraints
        self.cstr_manager.add("any_contacts", any_contacts, max_p=1.0)
        self.cstr_manager.add("abrupt_change_condition", abrupt_change_condition, max_p=1.0) #@#
        self.cstr_manager.add("stand_air_condition", stand_air_condition, max_p=1.0) #@#
        self.cstr_manager.add("position_protect", position_protect, max_p=1.0) #@#

        self.cstr_manager.log_all(self.episode_sums)

        # Timeout if episodes have reached their maximum duration
        timeout = self.progress_buf >= self.max_episode_length - 1

        # Get final termination probability for each env from all constraints
        self.cstr_prob = self.cstr_manager.get_probs()

        # Probability of termination used to affect the discounted sum of rewards
        ###self.reset_buf = self.cstr_prob # /!\ Must reactivate for CaT
        self.cat_cum_discount_factor *= 0.99 * (1 - self.cstr_prob)

        # Reset of environments upon timeout, invalid collision, being upside-down
        #if not self.allow_knee_contacts:
        #    self.reset_env_buf = timeout | cstr_base_contact | cstr_knee_contact | cstr_upsidedown
        #else:
        #    self.reset_env_buf = timeout | cstr_base_contact | cstr_upsidedown
        #self.reset_env_buf = timeout | cstr_upsidedown | cstr_termination_contacts | cstr_base_height_min #/!\#

        self.reset_env_buf = any_contacts
        self.reset_env_buf |= timeout
        self.reset_env_buf |= abrupt_change_condition #@#
        if self.pretrained:
            self.reset_env_buf |= position_protect #@#
            self.reset_env_buf |= not_stand
            self.reset_env_buf |= stand_air_condition #/!\#sit#
            self.reset_env_buf |= cstr_base_height_min
            #self.reset_env_buf |= not_stand_2

        self.reset_buf = self.reset_env_buf.clone() # /!\ Must deactivate for CaT

        self.extras["true_dones"] = self.reset_env_buf
        self.extras["dones"] = self.reset_env_buf.clone()
        self.extras["truncateds"] = timeout
        self.extras["raw_constraints"] = self.cstr_manager.get_raw_constraints()

    ####################
    # Other
    ####################

    def reset_idx(self, env_ids):
        """Reset environements when episodes have terminated."""

        self._randomize_dof_props(env_ids)
        self.gait_indices[env_ids] = 0

        # Randomize initial joint positions and velocities, as well as (x, y) position and yaw orientation of the base
        positions_offset = torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device)  # Multiplicative factor
        velocities = torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dof), device=self.device)
        yaw_offset = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)  # Already divided by 2 for next line
        xy_offset = torch_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device)
        quat_offset = torch.cat((torch.zeros((len(env_ids), 2), device=self.device), torch.sin(yaw_offset), torch.cos(yaw_offset)), dim=1)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset
        self.last_dof_pos[env_ids] = torch.unsqueeze(self.dof_pos[env_ids], 2)
        self.dof_vel[env_ids] = velocities

        # Set new root states
        if self.custom_origins:
            self.update_terrain_level(env_ids)
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += xy_offset
            self.root_states[env_ids, 3:7] = quat_offset
        else:
            self.root_states[env_ids] = self.base_init_state

        # Apply new root states to simulation
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # Apply new joint states to simulation
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # Refresh base tensors
        self.base_pos[env_ids] = self.root_states[env_ids, :3]
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]

        # Refresh base commands
        self.resample_commands(env_ids)

        # Log mean value of rewards over the terminated episodes
        if self.common_step_counter > 0 and self.numRewards > 0:
            self.rew_cum_reset[env_ids] = self.rew_mean[env_ids]
            self.rew_mean_reset[env_ids] = self.rew_mean[env_ids] / torch.maximum(self.progress_buf[env_ids].unsqueeze(0).transpose(0, 1), torch.ones((len(env_ids), 1),  dtype=torch.long, device=self.device))
            self.rew_mean[env_ids] = 0.0
            self.cat_discounted_cum_reward_reset[env_ids] = self.cat_discounted_cum_reward[env_ids]

        # Log mean value of constraints over the terminated episodes
        if self.common_step_counter > 0 and self.numConstraints > 0 and self.useConstraints in ["cat"]:
            self.cstr_mean_reset[env_ids] = self.cstr_mean[env_ids] / torch.maximum(self.progress_buf[env_ids].unsqueeze(0).transpose(0, 1), torch.ones((len(env_ids), 1),  dtype=torch.long, device=self.device))
            """s = ""
            v = ""
            for i, name in enumerate(self.cstr_names):
                s += "{:6s} | {:6.3f}\n".format(name, torch.mean(self.cstr_mean_reset[:, i]).item())
            print(s)"""
            self.cstr_mean[env_ids] = 0.0

        # Reset a bunch of state tensors
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.contacts_last[env_ids] = False
        self.contacts_filt[env_ids] = False
        self.contacts_touchdown[env_ids] = False
        self.feet_swing_time[env_ids] = 0.
        self.feet_swing_apex[env_ids] = 0.
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Fill extras and reset the cumulated reward sums of terminated episodes
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            akey = key if key.startswith("cstr_") else 'rew_' + key
            self.extras["episode"][akey] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        self.cat_cum_discount_factor[env_ids] = 1.
        self.cat_discounted_cum_reward[env_ids] = 0.

        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        self.extras["episode"]["terrain_level_max"] = torch.max(self.terrain_levels.float())
        self.extras["episode"]["gait_period"] = torch.max(self.feet_gait_time, dim=1).values
        self.extras["episode"]["foot_clearance"] = torch.max(self.feet_clearance, dim=1).values

    def resample_commands(self, env_ids):
        """Resample the base commands."""

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        #self.commands[env_ids] *= (torch.any(torch.abs(self.commands[env_ids, :3]) > self.vel_deadzone, dim=1)).unsqueeze(1)  # set small commands to zero

        # set small commands to zero
        #-#lin_cmd_cutoff = self.vel_deadzone
        #-#ang_cmd_cutoff = self.vel_deadzone
        #-#self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > lin_cmd_cutoff).unsqueeze(1)
        #-#self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > ang_cmd_cutoff)

    def update_terrain_level(self, env_ids):
        """Update terrain level of robots as they progress to put them in increasingly difficult terrains."""
        if not self.init_done or not self.curriculum:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # If robots went far enough during their episode, move them up.
        # This is monitored by move_up_flag that is set to True during the episode.
        # If robots did not move enough with respect to their velocity command, move them down.
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s * 0.25) * ~self.move_up_flag[env_ids]
        self.terrain_levels[env_ids] += 1 * self.move_up_flag[env_ids] - 1 * move_down
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

        # Reset flags
        self.move_up_flag[env_ids] = False

    def process_contacts(self):
        """Perform various processing linked to the contact status of the feet."""

        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contacts = self.contact_forces[:, self.grf_indices, 2] > 1.0
        # self.contacts_filt = torch.logical_or(contacts, self.contacts_last)
        # self.contacts_last = contacts
        self.contacts_filt = self.contact_forces[:, self.grf_indices, 2] > 1.0  # Or do not filter

        # True when touchdown for a given foot
        self.contacts_touchdown = (self.feet_swing_time > 0) * self.contacts_filt

        # Update swing time and apex height
        self.feet_swing_time += self.dt
        self.feet_swing_apex = torch.max(self.feet_swing_apex, self.get_feet_heights())

        # Keep in memory clearance of last swing phase
        self.feet_clearance[self.contacts_touchdown] = self.feet_swing_apex[self.contacts_touchdown]

        # Contact touchdown that are more filtered to be sure we only detect long swing phases
        filt_contacts_touchdown = torch.logical_and(self.feet_swing_time > 0.04, self.contacts_filt)
        if torch.any(filt_contacts_touchdown):
            self.feet_gait_time[filt_contacts_touchdown] = self.feet_swing_time[filt_contacts_touchdown] * 2

    def test_push_robots(self):
        base = torch.rand((self.num_envs, 2), device=self.device) * 2 * (0.4 - 0.15)
        self.root_states[:, 7:9] = torch.where(
            base < (0.4 - 0.15),
            base - 0.4,
            base - (0.4 - 0.15) + 0.15
        ) #torch_rand_float(-0.5, 0.5, (self.num_envs, 2), device=self.device) # lin vel x/y
        print(f"Pushing robot with value {self.root_states[:, 7:9]}")
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def push_robots(self):
        """Push the robots by assigning an instantaneous velocity to their base."""
        #p_push = self.dt / (self.max_episode_length_s * 2) # <- time step / (duration of X seconds * 2)
        p_push = self.dt / self.pushInterval_s # <- 1 / (number of steps)
        # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p)
        # The mean number of swaps for (1 / p) steps with probability p is 1. <- E[binomial(p, (1 / p))] = (1 / p) * p = 1
        push_idx = torch.bernoulli(torch.full((self.num_envs, ), p_push, device=self.device)).nonzero(as_tuple=False).flatten()
        base = torch.rand((len(push_idx), 2), device=self.device) * 2 * (0.4 - 0.15)
        self.root_states[push_idx, 7:9] = torch.where( #[-0.3, -0.15]U[0.15, 0.3]
            base < (0.4 - 0.15),
            base - 0.4,
            base - (0.4 - 0.15) + 0.15
        ) #torch_rand_float(-0.2, 0.2, (len(push_idx), 2), device=self.device) # lin vel x/y #/!\#
        #self.root_states[push_idx, 7:9] = torch_rand_float(-0.2, 0.2, (len(push_idx), 2), device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def pre_physics_step(self, actions):
        """Computing command torques with the PD controller and running the simulation steps between each call of the policy."""

        self.actions = actions.clone().to(self.device) #* 2. #/!\#
        self.q_diff_buf = torch.abs(self.default_dof_pos.to(self.device) + self.action_scale * actions.to(self.device) - self.dof_pos.to(self.device))
        #self.actions = actions.clone().detach().to(self.device) * self.action_scale_prime + self.action_bias
        ##self.actions = actions.clone().to(self.device)
        #self.actions[:, [0, 3, 6, 9]] *= 0.5 # scale down hip flexion range

        # If you want constant actions for debug purpose:
        # self.actions[:] = torch.tensor([0.2,  0.1679 , -0.37505, -0.2,  0.1679 , -0.37505, 0.2,  -0.1679 , 0.37505, -0.2,  -0.1679 , 0.37505])

        # There is self.decimation steps of simulation between each call to the policy
        # dt = 0.005
        # low level control at 0.005
        # hig level control at 0.2
        _switch = np.random.uniform()
        ratio_delay = (self.ratio_delay_end - self.ratio_delay_start) * (max((self.common_step_counter + 1)-self.ratio_delay_min_steps, 0) / (self.ratio_delay_max_steps-self.ratio_delay_min_steps)) + self.ratio_delay_start
        self.ratio_delay_log = ratio_delay

        self.lag_buffer = torch.cat([self.lag_buffer[:, :, 1:], actions.unsqueeze(dim=-1).clone()], dim=-1)
        is_swing = torch.tile((self.foot_positions[:, :, 2] > 0.0 + 0.023).unsqueeze(dim=-1), (1, 1, 3)).reshape((self.num_envs, self.num_dof))

        """self.lag_steps[:] = torch.from_numpy(np.random.randint(
            low=self.stance_lag_timesteps[0], 
            high=self.stance_lag_timesteps[1] + 1,
            size=self.lag_steps.shape
        )).to(self.device)
        self.lag_steps[is_swing] = torch.from_numpy(np.random.randint(
            low=self.swing_lag_timesteps[0], 
            high=self.swing_lag_timesteps[1] + 1,
            size=self.lag_steps[is_swing].shape
        )).to(self.device) # large delay when swing """ #/!\#
        cur_lag_steps = torch.minimum(self.valid_history_length, self.lag_steps) # min: 0, no delay; max: cfg.lag_timesteps
        # index: out[i][j] = input[i][j][index[i][j]]
        if self.common_step_counter >= self.lag_min_steps:
            used_actions = torch.gather(self.lag_buffer, 2, (self.lag_timesteps - cur_lag_steps).unsqueeze(2)).squeeze(-1)
        else:
            used_actions = actions
        self.valid_history_length = torch.clamp(cur_lag_steps + 1, max=self.lag_timesteps)

        if _switch > ratio_delay: #/!\#
            decimation = self.decimation
        else: #/!\#
            decimation = np.random.randint(self.decimation_range[0] + 1, self.decimation_range[1] + 1)
        for i in range(decimation):
            if self.use_actuator_net:
                self.joint_pos_err = self.dof_pos - (self.action_scale * used_actions + self.default_dof_pos) + self.motor_offsets
                self.joint_vel = self.dof_vel
                torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                                self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
                self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
                self.joint_pos_err_last = torch.clone(self.joint_pos_err)
                self.joint_vel_last_last = torch.clone(self.joint_vel_last)
                self.joint_vel_last = torch.clone(self.joint_vel)
            else:
                torques = torch.clip(
                    (
                        self.Kp
                        * (
                            self.action_scale * used_actions
                            + self.default_dof_pos
                            - self.dof_pos + self.motor_offsets
                        )
                        - self.Kd * self.dof_vel
                    ),
                    -100.0,  # Hard higher limit on torques
                    100.0,  # Hard lower limit on torques
                )

            torques = torques * self.motor_strengths
            if self.randomize_motor_friction:
                tau_sticktion = self.motor_Fs * torch.tanh(self.dof_vel / 0.1)
                tau_viscose = self.motor_mu_v * self.dof_vel
                torques -= tau_sticktion+tau_viscose

            # Saturating command torques (on Solo we saturate the max currents)
            # torques = torch.clamp(torques, -3.5, 3.5)

            # Logging for plotting purpose
            if self.debug_plots:
                self.log_dof_pos[self.progress_buf[0] * self.decimation + i] = self.dof_pos[0, :]
                self.log_dof_pos_cmd[self.progress_buf[0] * self.decimation + i] = self.action_scale * self.actions[0, :] + self.default_dof_pos[0, :]
                self.log_dof_vel[self.progress_buf[0] * self.decimation + i] = self.dof_vel[0, :]
                self.log_torques[self.progress_buf[0] * self.decimation + i] = torques[0, :]
                self.log_action_rate[self.progress_buf[0] * self.decimation + i] = (self.actions[0, :] - self.last_actions[0, :, 0]) / self.dt
                self.log_trajectory[self.progress_buf[0] * self.decimation + i, :7] = self.root_states[0, :7]
                self.log_trajectory[self.progress_buf[0] * self.decimation + i, 7:] = self.dof_pos[0, :]
                if self.common_step_counter > 1:
                    self.log_base_vel[self.progress_buf[0] * self.decimation + i, :3] = self.base_lin_vel[0, :]
                    self.log_base_vel[self.progress_buf[0] * self.decimation + i, 3:] = self.base_ang_vel[0, :]

            # Send desired joint torques to the simulation, run one step of simulator then refresh joint states
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torques))
            self.torques = torques.view(self.torques.shape)
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)

            # Gathering feet contact forces over several steps for averaging purpose (to avoid simulation glitches)
            self.filtered_contact_forces[:, :, :, 1:] = self.filtered_contact_forces[:, :, :, :-1]
            self.filtered_contact_forces[:, :, :, 0] = self.contact_forces[:, self.grf_indices, :]

            # Logging feet positions and velocities for plotting purpose
            if self.debug_plots:
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                self.log_feet_positions[self.progress_buf[0] * self.decimation + i] = self.rigid_body_state.view(
                    self.num_envs, self.num_bodies, 13
                )[0, self.feet_indices, 0:3]
                self.log_feet_velocities[self.progress_buf[0] * self.decimation + i] = self.rigid_body_state.view(
                    self.num_envs, self.num_bodies, 13
                )[0, self.feet_indices, 7:10]

        # Render the simulation (if there is a graphical interface)
        if self.force_render:
            self.render()

    def post_physics_step(self):
        """Updating tensors, computing rewards, constraints and observations after simulation steps."""

        # Updating all state tensors that are linked with the simulator
        # self.gym.refresh_dof_state_tensor(self.sim)  # done in pre_physics_step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)  # for imu force sensor

        # Increment counters
        self.progress_buf += 1  # Step counter for current episodes, set back to 0 upon env reset
        self.randomize_buf += 1  # Counter for domain randomization in VecTask (not really used)
        self.common_step_counter += 1  # Step counter since the start of the training/testing

        # Randomly push the robots
        if self.push_enable:
            if self.cfg["test"]:
                if self.common_step_counter == 250:
                    self.test_push_robots()
            else:
                self.push_robots()

        # Stop early if we are capturing a video
        if self.capturing_video and self.common_step_counter > 540:
            print("-- Stopping script early --")
            quit()

        # Update state quantities
        self.base_pos = self.root_states[:, :3]
        self.base_quat = self.root_states[:, 3:7]
        #self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10]) #/!\#
        self.base_lin_vel = quat_apply_yaw_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_forward_vec = quat_rotate_inverse(self.base_quat, self.forward_vec)
        self.init_feet_positions[self.progress_buf == 1] = self.foot_positions[self.progress_buf == 1]

        # center of mass
        rigid_body_pos = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:,:,0:3]
        com_pos = torch.sum(rigid_body_pos * self.mass.unsqueeze(-1),dim=1)/torch.sum(self.mass)
        self.com_pos = quat_rotate_inverse(self.root_states[:, 3:7], com_pos-self.root_states[:, 0:3])

        # Update leg phases
        phases_off = torch.tensor([0.0, torch.pi, torch.pi, 0.0], device=self.device)  # Phase offset between the legs
        self.phases = torch.tile(2 * torch.pi * self.phases_freq * self.progress_buf.unsqueeze(1) * self.dt, (1, 4)) + phases_off

        # Manually refresh foot position and velocities because taking the 0:3 splice breaks the automatic update
        self.foot_positions = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 7:10]
        self.foot_velocities_ang = self.rigid_body_state.view(
            self.num_envs, self.num_bodies, 13
        )[:, self.feet_indices, 10:13]

        self._step_contact_targets()

        # Update height scan
        self.measured_heights = self.get_heights()

        # Apply style constraints only on flat terrains
        self.is_flat_terrain = (self.measured_heights.var(1) < self.flat_terrain_threshold).float()

        # Update contact-related quantities
        self.process_contacts()

        # check if the robot went far enough to qualify for moving to higher difficulty
        if self.init_done and self.curriculum and self.custom_origins:
            # don't change on initial reset
            self.move_up_flag += torch.norm(self.root_states[:, :2] - self.env_origins[:, :2], dim=1) > (0.75 * 0.5 * self.terrain.env_length)

        # compute observations, rewards, resets, ...
        if self.useConstraints == "cat":
            self.compute_constraints_cat()

        if not self.diffReward:
            self.compute_reward_CaT()

        self.prev_commands = self.commands.clone()
        self.prev_foot_indices = self.foot_indices.clone()
        self.prev_feet_swing_time = self.feet_swing_time.clone()
        self.prev_contacts_touchdown = self.contacts_touchdown.clone()
        self.prev_foot_positions = self.foot_positions.clone()
        self.post_base_forward_vec = quat_apply(self.base_quat, self.forward_vec)
        self.prev_progress_buf = self.progress_buf.clone()
        self.prev_contact_forces = self.contact_forces.clone()

        self.prev_base_quat = self.base_quat.clone()

        # Display information in terminal for monitoring
        self.monitoring()

        # Set swing time and apex height to 0 for feet in contact (after reward/constraint computation)
        self.feet_swing_apex *= ~self.contacts_filt
        self.feet_swing_time *= ~self.contacts_filt

        # Retrieve environments that have to be reset
        if self.useConstraints == "cat":
            env_ids = self.reset_env_buf.nonzero(as_tuple=False).flatten()
        else:
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        self.compute_true_next_observations()

        # Keep history of previous actions, joint positions and velocities
        self.last_actions = torch.roll(self.last_actions, 1, 2)
        self.last_actions[:, :, 0] = self.actions[:]
        #self.last_dof_pos = torch.roll(self.last_dof_pos, 1, 2)
        #self.last_dof_pos[:, :, 0] = self.dof_pos[:]
        #self.last_dof_vel = torch.roll(self.last_dof_vel, 1, 2)
        #self.last_dof_vel[:, :, 0] = self.dof_vel[:]

        # Reset environments
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        # Refresh base velocity commands
        if self.useJoystick:
            # Get new velocity from Joystick
            self.joystick.update_v_ref(self.common_step_counter + self.decimation, 0)
            self.commands[0, 0] = self.joystick.v_ref[0, 0]
            self.commands[0, 1] = self.joystick.v_ref[1, 0]
            self.commands[0, 2] = self.joystick.v_ref[-1, 0]
            self.commands[0] *= torch.any(torch.abs(self.commands[0, :3]) > self.vel_deadzone) # set small commands to zero
        elif not self.cfg["env"]["onlyForwards"]:
            # Random velocity command resampling

            # The probabillity for an env that self.resample_commands produces
            # a null command is smaller than 2.2222%
            no_vel_command = torch.logical_and(
                torch.norm(self.commands[:, :2], dim=1) < self.vel_deadzone,
                torch.abs(self.commands[:, 2]) < self.vel_deadzone
            ).float()
            #-#p_resample_command = 0.01 * no_vel_command + (self.dt / self.max_episode_length_s) * (1 - no_vel_command)
            #-#resample_command_idx = torch.bernoulli(p_resample_command).nonzero(as_tuple=False).flatten()
            #-#if len(resample_command_idx) > 0:
            #-#    self.resample_commands(resample_command_idx)

            # Probability to have at least a zero command during entire trajectory is:
            # we ignore the dependency between having a zero commmand and the resampling probability
            # P(A U B) <= 2.2222% + (1-(1-0.02222/2000)^2000) - 2.2222% * (1-(1-0.02222/2000)^2000)
            # So the probability is less than 4.5%

            # Random angular velocity inversion during the episode to avoid having the robot moving in circle
            #-#p_ang_vel = self.dt / self.max_episode_length_s # <- time step / duration of X seconds
            # There will be a probability of 0.63 of having at least one swap after X seconds have elapsed
            # (1 / p) policy steps for X seconds, and the probability of having no swap at all is (1 - p)**(1 / p) = 0.37
            # The mean number of swaps for (1 / p) steps with probability p is 1.
            #-#self.commands[:, 2] *= 1 - 2 * torch.bernoulli(torch.full_like(self.commands[:, 2], p_ang_vel)).float()
            """resample_command_idx = (self.progress_buf % int(5. / self.dt)==0).nonzero(as_tuple=False).flatten()
            if len(resample_command_idx) > 0:
                self.resample_commands(resample_command_idx)"""

            # Now we increase the 4.5% to:
            # P(C U D) = 0.02222/2000 + 0.33333/2000 - P(C inter D) = 0.01%
            # P(A U B) <= 2.2222% + (1-(1-P(C U D))^2000) - 2.2222% * (1-(1-P(C U D))^2000) = 31.5%
            #-#p_zero_command = (1/3) * (self.dt / self.max_episode_length_s) * torch.ones_like(no_vel_command)
            #-#zero_command_idx = torch.bernoulli(p_zero_command).nonzero(as_tuple=False).flatten()
            #-#if len(zero_command_idx) > 0:
            #-#    self.commands[zero_command_idx] = 0.0

        # Compute observations
        self.update_depth_buffer()
        self.compute_observations()

        # Keep history of previous actions, joint positions and velocities
        ##self.last_actions = torch.roll(self.last_actions, 1, 2)
        ##self.last_actions[:, :, 0] = self.actions[:]
        self.last_dof_pos = torch.roll(self.last_dof_pos, 1, 2)
        self.last_dof_pos[:, :, 0] = self.dof_pos[:]
        self.last_dof_vel = torch.roll(self.last_dof_vel, 1, 2) #/!\# issue with reset
        self.last_dof_vel[:, :, 0] = self.dof_vel[:] #/!\# issue with reset

        # Logging for plotting purpose
        if self.debug_plots:
            self.log_feet_ctc_forces[self.progress_buf[0]] = self.contact_forces[0, self.grf_indices, :]
            if self.progress_buf[0] > (self.max_episode_length - 5) / 2:
                self.plot_logged_quantities()

        # If debug visualization is enabled, draw stuff in the simulation
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            # Draw small spheres to represent the height scan
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            sphere_geom = gymutil.WireframeSphereGeometry(0.01, 8, 8, None, color=(1, 0, 0))
            for i in range(self.num_envs):
                base_pos = (self.root_states[i, :3]).cpu().numpy()
                base_pos[2] = 0.1
                heights = self.measured_heights[i].cpu().numpy()
                height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
                for j in range(heights.shape[0]):
                    x = height_points[j, 0] + base_pos[0]
                    y = height_points[j, 1] + base_pos[1]
                    z = heights[j]
                    sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                    gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose)
                if i == 0:
                    self.camera_pos[:] = self.camera_pos[:] * 0.0 + base_pos[:] * 1.0
                    base_pos[:] = self.camera_pos[:]
                    cam_pos = gymapi.Vec3(base_pos[0] + 1.0, base_pos[1] - 0.7, base_pos[2] + 0.5)
                    cam_target = gymapi.Vec3(base_pos[0], base_pos[1], base_pos[2])
                    self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # If ground is flat
            if self.height_samples is None:
                return

            # Display a grid around the robot for easier visualization of terrain
            hsples = self.height_samples.cpu().numpy()
            gs = 48 # Grid size
            vs = self.terrain.vertical_scale
            hs = self.terrain.horizontal_scale
            base_pos = (self.root_states[0, :3]).cpu().numpy()
            bi = int(np.round((base_pos[0] + self.terrain.border_size) / hs))
            bj = int(np.round((base_pos[1] + self.terrain.border_size) / hs))

            x = np.linspace(bi - gs, bi + gs, 2 * gs + 1)
            y = np.linspace(bj - gs, bj + gs, 2 * gs + 1)
            grid_x, grid_y = (np.round(np.meshgrid(x, y))).astype(int)
            z = hsples[grid_x, grid_y]

            # Lines along X axis
            xx = (np.repeat(grid_x, 2, axis=1)[:, 1:-1]).reshape((-1, 2))
            yy = (np.repeat(grid_y, 2, axis=1)[:, 1:-1]).reshape((-1, 2))
            zz = (np.repeat(z, 2, axis=1)[:, 1:-1]).reshape((-1, 2))
            xyz = np.hstack((xx[:, 0:1], yy[:, 0:1], zz[:, 0:1], xx[:, 1:2], yy[:, 1:2], zz[:, 1:2]), dtype=np.float32) * np.array([hs, hs, vs] * 2, dtype=np.float32)
            xyz[:, [0, 1, 3, 4]] -= self.terrain.border_size
            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                xyz.shape[0],
                xyz,
                np.zeros((xyz.shape[0], 3), dtype=np.float32),
            )

            # Lines along Y axis
            xx = (np.repeat(grid_x, 2, axis=0)[1:-1, :]).transpose().reshape((-1, 2))
            yy = (np.repeat(grid_y, 2, axis=0)[1:-1, :]).transpose().reshape((-1, 2))
            zz = (np.repeat(z, 2, axis=0)[1:-1, :]).transpose().reshape((-1, 2))
            xyz = np.hstack((xx[:, 0:1], yy[:, 0:1], zz[:, 0:1], xx[:, 1:2], yy[:, 1:2], zz[:, 1:2]), dtype=np.float32) * np.array([hs, hs, vs] * 2, dtype=np.float32)
            xyz[:, [0, 1, 3, 4]] -= self.terrain.border_size
            self.gym.add_lines(
                self.viewer,
                self.envs[0],
                xyz.shape[0],
                xyz,
                np.zeros((xyz.shape[0], 3), dtype=np.float32),
            )

        self.dyn_obs_buf = self.obs_buf.clone()

    def monitoring(self):
        """Display information about rewards and constraints in terminal for monitoring."""

        # Initialize tensors for rewards
        if self.numRewards == -1:
            self.numRewards = len([item for item in list(self.episode_sums.keys()) if not item.startswith("cstr_")])
        if self.common_step_counter == 1:
            self.rew_mean = torch.zeros((self.num_envs, self.numRewards), dtype=torch.float, device=self.device, requires_grad=False)
            self.rew_cum_reset = torch.zeros((self.num_envs, self.numRewards), dtype=torch.float, device=self.device, requires_grad=False)
            self.rew_mean_reset = torch.zeros((self.num_envs, self.numRewards), dtype=torch.float, device=self.device, requires_grad=False)
            self.cat_discounted_cum_reward_reset = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device, requires_grad=False)
        # If there is any enabled rewards
        if self.numRewards > 0:
            for i, key in enumerate(list(self.episode_sums.keys())[:self.numRewards]):
                # episode_sums are already cumulated sums
                self.rew_mean[:, i] = self.episode_sums[key]

        # Initialize tensors for CaT
        if self.numConstraints == -1:
            self.numConstraints = len(self.cstr_manager.get_names())
        if self.common_step_counter == 1:
            self.cstr_mean = torch.zeros((self.num_envs, self.numConstraints), dtype=torch.float, device=self.device, requires_grad=False)
            self.cstr_mean_reset = torch.zeros((self.num_envs, self.numConstraints), dtype=torch.float, device=self.device, requires_grad=False)

        # If there is any enabled constraint, log violations
        if self.numConstraints > 0:
            if self.useConstraints == "cat":
                for i, key in enumerate(self.cstr_manager.get_names()):
                    self.cstr_mean[:, i] += self.cstr_manager.probs[key].max(1).values.gt(0.0).float()
            else:
                self.cstr_mean += self.cstr_buf

        if self.common_step_counter % self.cfg["horizon_length"] != 0:
            return

        if not self.cfg["test"]:
            print("------- Epoch {0} / {1}".format(int(self.common_step_counter / self.cfg["horizon_length"]) - 1, self.cfg["max_epochs"]))
        else:
            print("------- Testing")
        print(f"current ratio delay log: {self.ratio_delay_log:.3f}")
        print("")

        # Display average rewards and constraints violations over the last episodes
        table_rew = Texttable()
        table_rew.set_deco(Texttable.HEADER)
        table_rew.set_cols_align(["l", "r"])
        table_rew.set_cols_dtype(['t', 'f'])
        table_rew.add_rows(
            np.array([
                ["Rewards"] + list(self.episode_sums.keys())[:self.numRewards],
                ["Average/step"] + (torch.mean(self.rew_mean_reset, dim=0)).tolist()
            ]).transpose()
        )

        table_rew_cum = Texttable()
        table_rew_cum.set_deco(Texttable.HEADER)
        table_rew_cum.set_cols_align(["l", "r"])
        table_rew_cum.set_cols_dtype(['t', 'f'])
        table_rew_cum.add_rows(
            np.array([
                ["Info"] + ["Cumulated Rewards"] + ["Average level"] + ["Cumulated Discounted Rewards"],
                ["Value"] + [(torch.sum(torch.mean(self.rew_cum_reset, dim=0))).item()] + [self.terrain_levels.float().mean().item()] + \
                    [(torch.mean(self.cat_discounted_cum_reward_reset, dim=0)).item()]
            ]).transpose()
        )


        # Split the table strings into lines
        lines_A = table_rew.draw().split("\n") + ["", ""] + table_rew_cum.draw().split("\n") + [""]

        if self.numConstraints > 0:
            cstr_names = self.cstr_manager.get_names()

            table_cstr = Texttable()
            table_cstr.set_deco(Texttable.HEADER)
            table_cstr.set_cols_align(["l", "r"])
            table_cstr.set_cols_dtype(['t', 'f'])
            table_cstr.add_rows(
                np.array([
                    ["Constraints"] + cstr_names,
                    ["Violation (%)"] + (100.0 * torch.mean(self.cstr_mean_reset, dim=0)).tolist()
                ]).transpose()
            )

            # Split the table strings into lines
            lines_B = table_cstr.draw().split("\n") + [""]
        else:
            lines_B = [""]

        # Calculate the maximum and minimum width for each table
        max_width_A = max(len(line) for line in lines_A)
        max_width_B = max(len(line) for line in lines_B)

        # Line padding to bring them to the same length
        for i, line in enumerate(lines_A):
            lines_A[i] += ' ' * (max_width_A - len(line))

        # Add some padding between the tables
        padding = 5

        # Use itertools.zip_longest to handle tables with different numbers of lines
        filling = '' if len(lines_A) > len(lines_B) else ' ' * (max_width_A)
        for line_A, line_B in itertools.zip_longest(lines_A, lines_B, fillvalue=filling):
            print(f"{line_A}{' ' * padding}{line_B}")


    def plot_logged_quantities(self):
        """Display a few graphs to analyze results"""

        # Save figures to disk
        savefigs = True

        t_end = self.progress_buf[0].cpu().numpy() * self.dt

        N = 5
        from datetime import datetime
        date = datetime.now().strftime("_%m-%d_%H-%M-%S")

        from matplotlib import pyplot as plt

        # Display feet contact forces along Z in world frame
        plt.figure(figsize=(20, 12))
        lbl = ["FL", "FR", "HL", "HR"]
        for k in range(4):
            plt.plot(self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length - 2 * N),
                        self.log_feet_ctc_forces[N:-N, k, 2].cpu().numpy(), label=lbl[k])
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Contact force Z [N]")

        # Display feet velocities in world frame
        plt.figure(figsize=(20, 12))
        lbl = ["FL", "FR", "HL", "HR"]
        for k in range(4):
            plt.plot(self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                        self.log_feet_velocities[N:-N, k, 2].cpu().numpy(), label=lbl[k])
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Foot velocity [m/s]")

        # Display feet positions in world frame
        plt.figure(figsize=(20, 12))
        lbl = ["FL", "FR", "HL", "HR"]
        for k in range(4):
            plt.plot(self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                        self.log_feet_positions[N:-N, k, 2].cpu().numpy(), label=lbl[k])
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Foot position [m]")

        # Display base linear velocity in base frame
        lbl = ["Vx", "Vy", "Vz"]
        plt.figure(figsize=(20, 12))
        for k in range(3):
            plt.plot(self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                     self.log_base_vel[N:-N, k].cpu().numpy(), label=lbl[k])
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Linear velocity [m/s]")
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_lin_vel.png")

        # Display base angular velocity in base frame
        lbl = ["Wx", "Wy", "Wz"]
        plt.figure(figsize=(20, 12))
        for k in range(3):
            plt.plot(self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                     self.log_base_vel[N:-N, k+3].cpu().numpy(), label=lbl[k])
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Angular velocity [rad/s]")
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_ang_vel.png")

        # Display joint positions
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
        for i in range(12):
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                self.log_dof_pos[N:-N, i].cpu().numpy(),
                linestyle="-",
                label="Measured",
            )
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                self.log_dof_pos_cmd[N:-N, i].cpu().numpy(),
                linestyle="-",
                label="Desired",
            )

            axs[i % 3, int(i / 3)].set_title("Joint Positions")
            axs[i % 3, int(i / 3)].set_xlabel("Time [s]")
            axs[i % 3, int(i / 3)].set_ylabel(
                lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [rad]"
            )
            axs[i % 3, int(i / 3)].grid(True)
        plt.legend()
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_joint_pos.png")


        # Display joint velocities
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
        for i in range(12):
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                self.log_dof_vel[N:-N, i].cpu().numpy(),
                linestyle="-",
                label="Measured",
            )
            axs[i % 3, int(i / 3)].set_title("Joint Velocities")
            axs[i % 3, int(i / 3)].set_xlabel("Time [s]")
            axs[i % 3, int(i / 3)].set_ylabel(
                lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [rad/s]"
            )
            axs[i % 3, int(i / 3)].grid(True)
        plt.legend()
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_joint_vel.png")

        # Display joint torques
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
        for i in range(12):
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                (self.Kp * (self.log_dof_pos_cmd[N:-N, i] - self.log_dof_pos[N:-N, i])).cpu().numpy(),
                linestyle="-",
                label="P",
            )
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                (- self.Kd * self.log_dof_vel[N:-N, i]).cpu().numpy(),
                linestyle="-",
                label="D",
            )
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                (self.Kp * (self.log_dof_pos_cmd[N:-N, i] - self.log_dof_pos[N:-N, i]) - self.Kd * self.log_dof_vel[N:-N, i]).cpu().numpy(),
                linestyle="-",
                label="PD",
            )
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                (self.log_torques[N:-N, i]).cpu().numpy(),
                linestyle="-",
                label="Sent torques",
            )

            axs[i % 3, int(i / 3)].set_title("Joint Torques")
            axs[i % 3, int(i / 3)].set_xlabel("Time [s]")
            axs[i % 3, int(i / 3)].set_ylabel(
                lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [Nm]"
            )
            axs[i % 3, int(i / 3)].grid(True)
        plt.legend()
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_torques.png")

        # Display action rate
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
        for i in range(12):
            axs[i % 3, int(i / 3)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                (self.log_action_rate[N:-N, i]).cpu().numpy(),
                linestyle="-",
                label="Action rate",
            )

            axs[i % 3, int(i / 3)].set_title("Action rate")
            axs[i % 3, int(i / 3)].set_xlabel("Time [s]")
            axs[i % 3, int(i / 3)].set_ylabel(
                lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [rad/s]"
            )
            axs[i % 3, int(i / 3)].grid(True)
        plt.legend()
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_action_rate.png")

        # Display joint torques (only HFE and Knee of FR and HL legs)
        lgd1 = ["HFE", "Knee"]
        lgd2 = ["FR", "HL"]
        fig, axs = plt.subplots(2, 2, figsize=(20, 12), sharex=True)
        P = (self.Kp * (self.log_dof_pos_cmd[N:-N, [4, 5, 7, 8]] - self.log_dof_pos[N:-N, [4, 5, 7, 8]])).cpu().numpy()
        D = (- self.Kd * self.log_dof_vel[N:-N, [4, 5, 7, 8]]).cpu().numpy()
        PD = P + D
        for i in range(4):
            axs[i % 2, int(i / 2)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                P[:, i],
                linestyle="-",
                label="P",
            )
            axs[i % 2, int(i / 2)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                D[:, i],
                linestyle="-",
                label="D",
            )
            axs[i % 2, int(i / 2)].plot(
                self.dt * np.linspace(1 + N, self.max_episode_length - N, self.max_episode_length * self.decimation - 2 * N),
                PD[:, i],
                linestyle="-",
                label="PD",
            )

            axs[i % 2, int(i / 2)].set_title("Joint Torques")
            axs[i % 2, int(i / 2)].set_xlabel("Time [s]")
            axs[i % 2, int(i / 2)].set_ylabel(
                lgd1[i % 2] + " " + lgd2[int(i / 2)] + " [Nm]"
            )
            axs[i % 2, int(i / 2)].grid(True)
        plt.legend()
        plt.xlim([0.0, t_end])
        plt.grid(True)
        plt.tight_layout()
        if savefigs: plt.savefig(self.checkpoint_name + "_torques_duo.png")

        # Save base position and orientation in world frame
        np.save(self.checkpoint_name + "_trajectory.npy", self.log_trajectory.cpu().numpy())

        if self.debug_viz:
            plt.show(block=True)

        quit()

    def prepare_height_points(self):
        """Prepare the (x, y) grid of points to sample for the height scan"""

        step = self.cfg["env"]["learn"]["measured_points_step"]
        y = step * torch.tensor(self.cfg["env"]["learn"]["measured_points_y"], device=self.device, requires_grad=False)
        x = step * torch.tensor(self.cfg["env"]["learn"]["measured_points_x"], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        assert self.num_height_points == grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None):
        """Compute the height scan by retrieving the corresponding z for each (x, y) point of the grid scan"""

        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)
 
        points += self.terrain.border_size
        points = (points/self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        # First method
        # heights1 = self.height_samples[px, py]
        # heights2 = self.height_samples[px+1, py+1]
        # heights = torch.min(heights1, heights2)

        # Second method
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale

    def get_feet_heights(self):
        """Get feet heights with respect to the ground using ground heightmap"""

        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return self.foot_positions[:, :, -1]
        else:
            points = self.foot_positions.clone()
            points += self.terrain.border_size
            points = (points / self.terrain.horizontal_scale).long()
            points[:, :, 0] = torch.clip(
                points[:, :, 0], 0, self.height_samples.shape[0] - 2
            )
            points[:, :, 1] = torch.clip(
                points[:, :, 1], 0, self.height_samples.shape[1] - 2
            )

            heights1 = self.height_samples[points[:, :, 0], points[:, :, 1]]
            heights2 = self.height_samples[points[:, :, 0] + 1, points[:, :, 1]]
            heights3 = self.height_samples[points[:, :, 0], points[:, :, 1] + 1]
            heights = torch.min(heights1, heights2)
            heights = torch.min(heights, heights3)

            return self.foot_positions[:, :, -1] - heights * self.terrain.vertical_scale

    def _diff_reward_raibert_heuristic(self, full_obs):
        commands = self.prev_commands
        footsteps_in_body_frame = full_obs[:, -23:-15].reshape(-1, 4, 2)

        # nominal positions: [FR, FL, RR, RL]
        desired_stance_width = 0.25 # 0.125
        desired_ys_nom = torch.tensor([desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=self.device).unsqueeze(0)

        desired_stance_length = 0.45 # 0.225
        desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=self.device).unsqueeze(0)

        # raibert offsets
        phases = torch.abs(1.0 - (self.prev_foot_indices * 2.0)) * 1.0 - 0.5
        frequencies = torch.tensor([2.0], device=self.device)
        x_vel_des = commands[:, 0:1]
        yaw_vel_des = commands[:, 2:3]
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

    def diffRecalculateReward(self, full_obs, actions, last_actions, last_last_actions, imagined_trajs = None, maac = False):
        allow_contact_steps = 30
        scale_factor_low = 0.33 # 0.30 # 0.33 #/!\#sit#
        scale_factor_high = 0.45
        allow_lin_vel_xy_steps = 60
        condition_lin_vel = self.prev_progress_buf > allow_lin_vel_xy_steps

        # Velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.prev_commands[:, :2] - full_obs[:, :2] / self.lin_vel_scale), dim=1)
        ang_vel_error = torch.square(self.prev_commands[:, 2] - full_obs[:, 5] / self.ang_vel_scale)
        rew_lin_vel_xy = torch.exp(-lin_vel_error / self.lin_vel_delta) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / self.ang_vel_delta) * self.rew_scales["ang_vel_z"]

        actual_lin_vel = full_obs[:, :3] / self.lin_vel_scale
        lin_vel_error = torch.sum(torch.square(self.prev_commands[:, :2] - actual_lin_vel[:, :2]), dim=1)
        rew_lin_vel_xy = torch.exp(-lin_vel_error/self.lin_vel_delta) # self.cfg.rewards.tracking_sigma)
        forward = quat_apply(self.prev_base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.prev_base_quat, self.upright_vec)
        is_stand = (torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)) > 0.9
        scaling_factor = (torch.clip(
            full_obs[:, -1], min=scale_factor_low, max=scale_factor_high
        ) - scale_factor_low) / (scale_factor_high - scale_factor_low)
        rew_lin_vel_xy = rew_lin_vel_xy * is_stand.float() * scaling_factor
        rew_lin_vel_xy *= self.rew_scales["lin_vel_xy"]
        rew_lin_vel_xy_ = rew_lin_vel_xy.clone()

        rew_lin_vel_xy = torch.square(actual_lin_vel[:, :2]).sum(-1)
        rew_lin_vel_xy = rew_lin_vel_xy * condition_lin_vel.float()
        rew_lin_vel_xy *= -0.4 # -0.2
        rew_lin_vel_xy += rew_lin_vel_xy_

        """heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        heading_error = torch.square(wrap_to_pi(self.commands[:, 2] - heading) / np.pi)
        heading_rew = torch.exp(-heading_error / self.lin_vel_delta)
        heading_rew = heading_rew * is_stand.float() * scaling_factor
        heading_rew *= self.rew_scales["ang_vel_z"]
        rew_ang_vel_z = heading_rew"""

        rew_action_rate = torch.sum(torch.square(actions - last_actions), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]
        rew_action_rate += torch.sum(torch.square(actions - 2 * last_actions + last_last_actions), dim=1) * (self.action_scale ** 2) * self.rew_scales["action_rate"]

        rew_orient = torch.sum(torch.square(full_obs[:, 9:11]), dim=1) * self.rew_scales["orient"]
        rew_lin_vel_z = torch.square(full_obs[:, 2] / self.lin_vel_scale) * self.rew_scales["lin_vel_z"]

        #forward_vec_obs#rew_dof_vel = torch.sum(torch.square(full_obs[:, 27:39] / self.dof_vel_scale), dim=1) * self.rew_scales["dof_vel"]
        rew_dof_vel = torch.sum(torch.square(full_obs[:, 24:36] / self.dof_vel_scale), dim=1) * self.rew_scales["dof_vel"] #no_forward_vec_obs#

        rew_torque = torch.sum(torch.square(full_obs[:, -27:-15] / self.torques_scale), dim=1) * self.rew_scales["torque"]
        rew_torque_limits = torch.sum((torch.abs(full_obs[:, -27:-15] / self.torques_scale) - self.torque_limits * self.soft_torque_limit).clip(min=0.), dim=1) * self.rew_scales["torque_limits"]
        rew_ang_vel_xy = torch.sum(torch.square(full_obs[:, 3:5] / self.ang_vel_scale), dim=1) * self.rew_scales["ang_vel_xy"]

        self.diff_reward_rew_action_rate = rew_action_rate.clone()

        condition = self.prev_progress_buf > allow_contact_steps
        forward = full_obs[:, -7:-4]
        upright_vec = full_obs[:, -4:-1]

        rew_ang_vel_z = torch.square(full_obs[:, 5] / self.ang_vel_scale) * -0.1 * condition.float() # -0.25
        #rew_ang_vel_z = torch.square(full_obs[:, 5] / self.ang_vel_scale) * -0.1 * is_stand.float() # -0.25

        phases = 1 - torch.abs(1.0 - torch.clip((self.prev_foot_indices[:, self.fidx] * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (full_obs[:, -11:-7][:, self.fidx]).view(self.num_envs, -1) # - reference_heights
        target_height = 0.08 * phases + 0.02 # 0.08 * phases + 0.02 # offset for foot radius 2cm
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states[:, self.fidx])
        #rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1) * is_stand.float() * condition_lin_vel.float() * self.rew_scales["feetClearance"]
        rew_feet_clearance = torch.sum(rew_foot_clearance, dim=1) * condition.float() * self.rew_scales["feetClearance"]

        rew_foot2contact = - torch.abs((self.contact_forces[:, self.grf_indices, 2] > 1.0).sum(1) - 2) / 2 * self.rew_scales["foot2contact"]

        foot_forces = full_obs[:, -15:-11] / self.foot_forces_scale
        desired_contact = self.desired_contact_states
        rew_tracking_contacts_shaped_force = 0.
        #for i in range(4):
        for i in self.fidx:
            rew_tracking_contacts_shaped_force += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / 100.))
        #rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force * is_stand.float() * condition_lin_vel.float() * self.rew_scales["trackingContactsShapedForce"] / 4
        rew_tracking_contacts_shaped_force = rew_tracking_contacts_shaped_force * is_stand.float() * self.rew_scales["trackingContactsShapedForce"] / 4

        root_height = full_obs[:, -1]
        lift_up_threshold = [0.27, 0.55] # [0.20, 0.55] # [0.27, 0.55] #/!\#sit#
        rew_lift_up = (root_height - lift_up_threshold[0]) / (lift_up_threshold[1] - lift_up_threshold[0])
        rew_lift_up = torch.clamp(rew_lift_up, 0., 1.) * 0.5

        rew_fall = (lift_up_threshold[0] - root_height).clip(min=0.) * (-1.)

        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
        rew_upright = torch.square(0.5 * cosine_dist + 0.5) * 1.0

        contact = self.prev_contact_forces[:, self.feet_indices[self.fidx], 2] < 1.
        calf_contact = self.contact_forces[:, self.knee_indices[self.fidx], 2] < 1.
        unhealthy_condition = torch.logical_and(~calf_contact, contact)
        rew_rear_air = (torch.all(contact, dim=1).float() + unhealthy_condition.sum(dim=-1).float()) * (-0.5)

        rew_stand_air = (full_obs[:, -11:-7][:, self.fidx] - 0.06).clip(min=0.0).sum(-1) * (-40.0)
        rew_stand_air += (full_obs[:, -11:-7][:, [2, 3]]).clip(max=0.06).sum(-1) * (5.0)
        rew_stand_air *= (~condition).float() #/!\#/?\#

        #rew_back_still = torch.sum(torch.square(full_obs[:, 27:39][:, 6:] / self.dof_vel_scale), dim=1) * is_stand.float() * condition_lin_vel.float() * 0. # (-0.01) #forward_vec_obs#

        dof_pos_lower = (self.dof_pos_limits[:, 0] - full_obs[:, 12:24] / self.dof_pos_scale).clip(min=0.)
        dof_pos_upper = (full_obs[:, 12:24] / self.dof_pos_scale - self.dof_pos_limits[:, 1]).clip(min=0.)
        rew_position_protect = (dof_pos_lower.sum(-1) + dof_pos_upper.sum(-1)) * (-100.)

        foot_positions = full_obs[:, -39:-27].reshape(-1, 4, 3)
        desired_foot_positions = torch.clone(self.init_feet_positions[:, self.fidx])
        desired_foot_positions[:, :, 2] = 0.02

        front_foot_shift = torch.abs(foot_positions[:, self.fidx, 1] - desired_foot_positions[:, :, 1]).mean(dim=1)
        """front_foot_shift = torch.norm(foot_positions[:, self.fidx] - desired_foot_positions, dim=-1).mean(dim=1)
        init_rfoot_positions = torch.clone(self.init_feet_positions[:, [2, 3]])
        rear_foot_shift = torch.norm( torch.stack([
            (foot_positions[:, [2, 3], 0] - init_rfoot_positions[:, :, 0]).clamp(min=0), #/!\#
            torch.abs(init_rfoot_positions[:, :, 1] - foot_positions[:, [2, 3], 1])
        ], dim=-1), dim=-1).mean(dim=1)"""
        #rew_foot_shift = (front_foot_shift + rear_foot_shift) * (~condition).float() * self.rew_scales["foot_shift"]
        rew_foot_shift = front_foot_shift * (~condition).float() * self.rew_scales["foot_shift"]

        dof_pos = full_obs[:, 12:24]
        rew_front_symmetry = torch.square(dof_pos[:, 0] + dof_pos[:, 3]) #/!\#
        rew_front_symmetry += torch.square(dof_pos[:, 1] - dof_pos[:, 4])
        rew_front_symmetry += torch.square(dof_pos[:, 2] - dof_pos[:, 5])
        rew_rear_symmetry = torch.square(dof_pos[:, 6] + dof_pos[:, 9]) #/!\#
        rew_rear_symmetry += torch.square(dof_pos[:, 7] - dof_pos[:, 10])
        rew_rear_symmetry += torch.square(dof_pos[:, 8] - dof_pos[:, 11])
        #rew_symmetry = (rew_front_symmetry * (~condition).float() + rew_rear_symmetry) * -0.1 # -0.1
        rew_symmetry = (rew_rear_symmetry) * -0.0 # -0.1

        slip_condition = self.prev_foot_positions[:, :, 2] < 0.03
        # xy lin vel
        foot_velocities = torch.square(torch.norm(full_obs[:, -39:-31].reshape(-1, 4, 2) / self.foot_velocity_scale, dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(full_obs[:, -31:-27].reshape(-1, 4, 1) / (np.pi * self.foot_ang_velocity_scale), dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(slip_condition.float() * (foot_velocities + foot_ang_velocities), dim=1) * (-0.04 * 0.1)

        diff_rew = rew_upright + rew_lift_up + rew_stand_air + rew_lin_vel_xy + rew_feet_clearance + rew_tracking_contacts_shaped_force + rew_torque_limits + rew_ang_vel_z + rew_action_rate + rew_dof_vel
        diff_rew = torch.clip(diff_rew, min=0., max=None)

        return diff_rew

    def full2partial_state(self, full_obs):
        if self.add_noise: #/!\#
            obs_meas = full_obs + (2 * torch.rand_like(full_obs) - 1) * self.noise_scale_vec #/!\#
        else:
            obs_meas = full_obs
        return obs_meas[:, 3:-27] # full_obs[:, 3:-39] # full_obs[:, 3:-27] # full_obs[:, 3:-15] #obs_changes#

    def rebuild_output_dyn(self, x, a):
        return torch.cat([
            x[..., :6],
            self.prev_commands[..., :3].clone(),
            x[..., 6:33],
            a.clone(),
            x[..., 33:]
        ], dim = -1)

    def debuild_output_dyn(self, x):
        # 3, 12, 12, 12 <-
        return torch.cat([
            x[..., :6],
            x[..., 9:36],
            x[..., 48:],
        ], dim = -1)

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

def quat_apply_yaw_inverse(quat, vec):
    quat_yaw_inverse = quat.clone().view(-1, 4)
    quat_yaw_inverse[:, :2] = 0.
    quat_yaw_inverse[:, 2] = -1 * quat_yaw_inverse[:, 2]
    quat_yaw_inverse = normalize(quat_yaw_inverse)
    return quat_apply(quat_yaw_inverse, vec)

def normalize_range(x: torch.Tensor, limit):
    if isinstance(limit[0], list):
        low = torch.from_numpy(np.array(limit[0])).to(x.device)
        high = torch.from_numpy(np.array(limit[1])).to(x.device)
    else:
        low = limit[0]
        high = limit[1]
    mean = (low + high) / 2
    scale = (high - low) / 2
    if isinstance(scale, torch.Tensor):
        scale = torch.clamp(scale, min=1e-5)
    else:
        scale = max(scale, 1e-5)
    return (x - mean) / scale
