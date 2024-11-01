# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import numpy as np
import torch

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from omni.isaac.lab.envs import ManagerBasedRLEnv

import sys
from os.path import abspath, dirname, join
sys.path.append(abspath(join(dirname(abspath(__file__)), "../../../")))

from source.assets.go1_cfg import GO1_CFG

from source.torch_utils import normalize, quat_apply

#@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

#@ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

gaits = {"pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5]}

dt = 4
durations = 0.5
kappa_gait_probs = 0.07

step_frequency_cmd = 3.0
frequencies = step_frequency_cmd

gait_indices = None
gait_force_sigma = 100
gait_vel_sigma = 10

desired_contact_states = None

clock_inputs = None
doubletime_clock_inputs = None
halftime_clock_inputs = None

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = GO1_CFG
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

def get_gait_indices(env):
    global gait_indices
    if gait_indices is None:
        gait_indices = torch.zeros(env.num_envs,device=env.device)
    return gait_indices

def get_clock_inputs(env):
    global clock_inputs
    if clock_inputs is None:
        clock_inputs = torch.zeros((env.num_envs,4),device=env.device)
    global doubletime_clock_inputs
    if doubletime_clock_inputs is None:
        doubletime_clock_inputs = torch.zeros((env.num_envs,4),device=env.device)
    global halftime_clock_inputs
    if halftime_clock_inputs is None:
        halftime_clock_inputs = torch.zeros((env.num_envs,4),device=env.device)
    return clock_inputs, doubletime_clock_inputs, halftime_clock_inputs

def get_desired_contact_states(env):
    global desired_contact_states
    if desired_contact_states is None:
        desired_contact_states = torch.zeros((env.num_envs,4),device=env.device)
    return desired_contact_states

def get_gait_force_sigma():
    global gait_force_sigma
    return gait_force_sigma

def get_gait_vel_sigma():
    global gait_vel_sigma
    return gait_vel_sigma

def init_desired_contact_states(env):
    gait_indices = get_gait_indices(env)
    gait = torch.tensor(gaits["trotting"], device=env.device)

    global dt, durations, kappa_gait_probs, step_frequency_cmd

    phases = gait[0]
    offsets = gait[1]
    bounds = gait[2]
    gait_indices = torch.remainder(gait_indices + dt * frequencies, 1.0)
    clock_inputs, doubletime_clock_inputs, halftime_clock_inputs = get_clock_inputs(env)

    # Starting the calculations
    foot_indices = [gait_indices + phases + offsets + bounds,
                    gait_indices + offsets,
                    gait_indices + bounds,
                    gait_indices + phases]

    foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

    for idxs in foot_indices:
        stance_idxs = torch.remainder(idxs, 1) < durations
        swing_idxs = torch.remainder(idxs, 1) > durations

        idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations)
        idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations) * (
                    0.5 / (1 - durations))
        
    clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[:,0])
    clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[:,1])
    clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[:,2])
    clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[:,3])

    doubletime_clock_inputs[:, 0] = torch.sin(4 * np.pi * foot_indices[:,0])
    doubletime_clock_inputs[:, 1] = torch.sin(4 * np.pi * foot_indices[:,1])
    doubletime_clock_inputs[:, 2] = torch.sin(4 * np.pi * foot_indices[:,2])
    doubletime_clock_inputs[:, 3] = torch.sin(4 * np.pi * foot_indices[:,3])

    halftime_clock_inputs[:, 0] = torch.sin(np.pi * foot_indices[:,0])
    halftime_clock_inputs[:, 1] = torch.sin(np.pi * foot_indices[:,1])
    halftime_clock_inputs[:, 2] = torch.sin(np.pi * foot_indices[:,2])
    halftime_clock_inputs[:, 3] = torch.sin(np.pi * foot_indices[:,3])

    # von mises distribution
    kappa = kappa_gait_probs
    smoothing_cdf_start = torch.distributions.normal.Normal(0, kappa).cdf
      # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

    smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[:,0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[:,0], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[:,0], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[:,0], 1.0) - 0.5 - 1)))
    smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[:,1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[:,1], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[:,1], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[:,1], 1.0) - 0.5 - 1)))
    smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[:,2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[:,2], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[:,2], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[:,2], 1.0) - 0.5 - 1)))
    smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[:,3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[:,3], 1.0) - 0.5)) +
                                smoothing_cdf_start(torch.remainder(foot_indices[:,3], 1.0) - 1) * (
                                        1 - smoothing_cdf_start(
                                    torch.remainder(foot_indices[:,3], 1.0) - 0.5 - 1)))

    desired_contact_states = get_desired_contact_states(env)
    desired_contact_states[:, 0] = smoothing_multiplier_FL
    desired_contact_states[:, 1] = smoothing_multiplier_FR
    desired_contact_states[:, 2] = smoothing_multiplier_RL
    desired_contact_states[:, 3] = smoothing_multiplier_RR



















def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.track_lin_vel_xy_exp(env,std,command_name,asset_cfg)
    return reward

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.track_ang_vel_z_exp(env,std,command_name,asset_cfg)
    return reward

def feet_contact_forces(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # Getting global values and parameters for the evaluation
    global desired_contact_states
    if desired_contact_states is None:
        init_desired_contact_states(env)

    std = 0.25

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    foot_contact_forces = net_contact_forces[:, :, sensor_cfg.body_ids]
    foot_forces = torch.sum(torch.norm(foot_contact_forces, dim=-1),dim=1)
    
    gait_force_sigma = get_gait_force_sigma()

    reward = 0
    for i in range(4):
        reward += (1 - desired_contact_states[:, i]) * \
                    (torch.exp(-1 * (foot_forces[:, i] ** 2) / gait_force_sigma))
    return torch.mul(reward, std)

def feet_contact_vel(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # Getting global values and parameters for the evaluation
    global desired_contact_states
    if desired_contact_states is None:
        init_desired_contact_states(env)

    std = 0.25

    asset: RigidObject = env.scene[asset_cfg.name]
    foot_velocities = \
        torch.norm(\
            asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]\
        ,dim=2).view(env.num_envs, -1)
    
    gait_vel_sigma = get_gait_vel_sigma()

    reward = 0
    for i in range(4):
        reward += (desired_contact_states[:, i] * \
                     (torch.exp(-1 * (foot_velocities[:, i] ** 2) / gait_vel_sigma)))
    return torch.mul(reward, std)

def base_height_l2(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    # TODO: Fix this for rough-terrain.
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)

def flat_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.flat_orientation_l2(env,asset_cfg)
    return reward

def raibert_footswing(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)

    asset: RigidObject = env.scene[asset_cfg.name]
    base_assets_pos = asset.data.root_pos_w[:]
    base_assets_quat = asset.data.root_quat_w[:]
    foot_assets_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]

    cur_footsteps_translated = foot_assets_pos - base_assets_pos.unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)

    conjugate_base_quat = quat_conjugate(base_assets_quat)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = quat_apply_yaw(\
            conjugate_base_quat, cur_footsteps_translated[:, i, :])
        
    # nominal positions: [FR, FL, RR, RL]
    desired_stance_width = 0.3
    desired_ys_nom = torch.tensor([desired_stance_width / 2,  -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2], device=env.device).unsqueeze(0)

    desired_stance_length = 0.45
    desired_xs_nom = torch.tensor([desired_stance_length / 2,  desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2], device=env.device).unsqueeze(0)

    # raibert offsets
    global gaits, step_frequency_cmd
    gait = torch.tensor(gaits["trotting"], device=env.device)
    phase = gait[0]
    offsets = gait[1]
    bounds = gait[2]
    dt = 4
    frequencies = step_frequency_cmd

    gait_indices = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    gait_indices = torch.remainder(gait_indices + dt * frequencies, 1.0)

    foot_indices = [gait_indices + phase + offsets + bounds,
                    gait_indices + bounds,
                    gait_indices + offsets,
                    gait_indices + phase]

    foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)
    phases = torch.abs(1.0 - (foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = torch.tensor(step_frequency_cmd,device=env.device)

    x_vel_des = commands[:,0:1]
    yaw_vel_des = commands[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2
    
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies)
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies)

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset

    desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)

    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])

    reward = torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))
    return reward

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    global desired_contact_states
    if desired_contact_states is None:
        init_desired_contact_states(env)

    global gaits, step_frequency_cmd
    gait = torch.tensor(gaits["trotting"], device=env.device)
    phase = gait[0]
    offsets = gait[1]
    bounds = gait[2]
    dt = 4
    frequencies = step_frequency_cmd

    gait_indices = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    gait_indices = torch.remainder(gait_indices + dt * frequencies, 1.0)

    foot_indices = [gait_indices + phase + offsets + bounds,
                    gait_indices + bounds,
                    gait_indices + offsets,
                    gait_indices + phase]
    asset: RigidObject = env.scene[asset_cfg.name]

    foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

    phases = 1 - torch.abs(1.0 - torch.clip((foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)

    target_heights = torch.full_like(phases,target_height)
    target_heights = target_heights * phases + 0.02
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_heights)
    reward = foot_z_target_error * (1 - desired_contact_states)
    reward = torch.sum(reward, dim=1)
    return reward

def lin_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.lin_vel_z_l2(env,asset_cfg)
    return reward

def ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.ang_vel_xy_l2(env,asset_cfg)
    return reward

def feet_slip(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(torch.square(body_vel.norm(dim=-1) * contacts), dim=1)
    return reward

def undesired_contacts(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    reward = mdp.undesired_contacts(env,threshold,sensor_cfg)
    return reward

def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.joint_pos_limits(env,asset_cfg)
    return reward

def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.joint_torques_l2(env,asset_cfg)
    return reward

def joint_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.joint_vel_l2(env,asset_cfg)
    return reward

def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reward = mdp.joint_acc_l2(env,asset_cfg)
    return reward

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    reward = mdp.action_rate_l2(env)
    return reward

def action_rate_2nd_order_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward = torch.sum(torch.square(env.action_manager.prev_prev_action - \
    2*env.action_manager.prev_action + env.action_manager.action), dim=1)
    return reward

def jump(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    reference_heights = 0
    base_height_target = 0.34
    height_cmd = 0.0

    asset: RigidObject = env.scene[asset_cfg.name]
    body_height = asset.data.root_pos_w[:, 2] - reference_heights

    jump_height_target =  height_cmd + base_height_target
    reward = - torch.square(body_height - jump_height_target)
    return reward
















@configclass
class RewardsCfg:

    """Reward terms for the MDP."""
    # -- TASK
    # xy velocity traking
    track_lin_vel_xy_exp = RewTerm(
        func=track_lin_vel_xy_exp, weight=0.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},)
    # yaw velocity traking
    track_ang_vel_z_exp = RewTerm(
        func=track_ang_vel_z_exp, weight=0.1,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},)
    # z velocity traking
    #track_lin_vel_z = RewTerm(
    #    func=mdp.lin_vel_z_l2, weight=-0.004)
    # ang velocity traking xy
    #track_ang_vel_xy = RewTerm(
    #    func=mdp.ang_vel_xy_l2, weight=-2e-5)
    # -- AUGMENTED AUXILIARY
    # swing phase traking (force)
    contact_forces = RewTerm(
        func=feet_contact_forces, weight=0.08,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
    # stance phase traking (velocity)
    contact_vel = RewTerm(
        func=feet_contact_vel, weight=0.08,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})
    # body height tracking
    base_height_l2 = RewTerm(
        func=base_height_l2, weight=-0.05,
        params={"target_height": 0.3},)
    # body pitch traking
    flat_orientation_l2 = RewTerm(
        func=flat_orientation_l2, weight=-0.1)
    # raibert heuristic footswing tracking
    raibert_footswing = RewTerm(
        func=raibert_footswing, weight=-0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})
    # footswing height tracking 
    footswing = RewTerm(
        func=foot_clearance_reward,weight=-0.6,
        params={
            "std": math.sqrt(0.25),
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),},)
    # z velocity
    lin_vel_z_l2 = RewTerm(
        func=lin_vel_z_l2, weight=-0.2) #-4e-4)
    # roll-pitch velocity
    ang_vel_xy_l2 = RewTerm(
        func=ang_vel_xy_l2, weight=-2e-5)
    # foot slip
    feet_slide = RewTerm(
        func=feet_slip, weight=-8e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },)
    # thigh/calf collision
    thigh_collision = RewTerm(
        func=undesired_contacts, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
            "threshold": 1.0,},)
    calf_collision = RewTerm(
        func=undesired_contacts, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            "threshold": 1.0,},)
    # joint limit violation
    dof_pos_limits = RewTerm(
        func=joint_pos_limits, weight=-0.2)
    # joint torques
    dof_torques_l2 = RewTerm(
        func=joint_torques_l2, weight=-2e-6)
    # joint velocities
    dof_vel_l2 = RewTerm(
        func=joint_vel_l2, weight=-2e-6)
    # joint accelerations 
    dof_acc_l2 = RewTerm(
        func=joint_acc_l2, weight=-5e-9)
    # action smoothing
    action_rate_l2 = RewTerm(
        func=action_rate_l2, weight=-2e-3)
    # action smoothing, 2nd order
    action_rate_2nd_order_l2 = RewTerm(
        func=action_rate_2nd_order_l2, weight=-2e-3)
    # jump
    #action_rate_2nd_order_l2 = RewTerm(
    #    func=jump, weight=0.2)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


@configclass
class Go1FlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=25)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt


class Go1FlatEnvCfg_PLAY(Go1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None
