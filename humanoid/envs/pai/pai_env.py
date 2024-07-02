# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi

import torch
from humanoid.envs import LeggedRobot

from humanoid.utils.terrain import  HumanoidTerrain
# from collections import deque


class PaiFreeEnv(LeggedRobot):
    '''
    PaiFreeEnv is a class that represents a custom environment for a legged robot.

    Args:
        cfg (LeggedRobotCfg): Configuration object for the legged robot.
        sim_params: Parameters for the simulation.
        physics_engine: Physics engine used in the simulation.
        sim_device: Device used for the simulation.
        headless: Flag indicating whether the simulation should be run in headless mode.

    Attributes:
        last_feet_z (float): The z-coordinate of the last feet position.
        feet_height (torch.Tensor): Tensor representing the height of the feet.
        sim (gymtorch.GymSim): The simulation object.
        terrain (HumanoidTerrain): The terrain object.
        up_axis_idx (int): The index representing the up axis.
        command_input (torch.Tensor): Tensor representing the command input.
        privileged_obs_buf (torch.Tensor): Tensor representing the privileged observations buffer.
        obs_buf (torch.Tensor): Tensor representing the observations buffer.
        obs_history (collections.deque): Deque containing the history of observations.
        critic_history (collections.deque): Deque containing the history of critic observations.

    Methods:
        _push_robots(): Randomly pushes the robots by setting a randomized base velocity.
        _get_phase(): Calculates the phase of the gait cycle.
        _get_gait_phase(): Calculates the gait phase.
        compute_ref_state(): Computes the reference state.
        create_sim(): Creates the simulation, terrain, and environments.
        _get_noise_scale_vec(cfg): Sets a vector used to scale the noise added to the observations.
        step(actions): Performs a simulation step with the given actions.
        compute_observations(): Computes the observations.
        reset_idx(env_ids): Resets the environment for the specified environment IDs.
    '''
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.last_feet_z = 0.05
        self.feet_height = torch.zeros((self.num_envs, 2), device=self.device)
        self.reset_idx(torch.tensor(range(self.num_envs), device=self.device))
        self.compute_observations()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
            随机推机器人，通过设置随机的基础速度来模拟冲击。
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        max_push_angular = self.cfg.domain_rand.max_push_ang_vel
        self.rand_push_force[:, :2] = torch_rand_float(
            -max_vel, max_vel, (self.num_envs, 2), device=self.device)  # lin vel x/y
        self.root_states[:, 7:9] = self.rand_push_force[:, :2]

        self.rand_push_torque = torch_rand_float(
            -max_push_angular, max_push_angular, (self.num_envs, 3), device=self.device)

        self.root_states[:, 10:13] = self.rand_push_torque

        self.gym.set_actor_root_state_tensor(
            self.sim, gymtorch.unwrap_tensor(self.root_states))

    def check_termination(self):
        """ Check if environments need to be reset
        检查环境是否需要重置。它通过检查接触力、时间和基础位置来确定是否需要重置环境。
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= torch.any(self.base_pos[:, 2:3] < 0.15, dim=1)
        self.reset_buf |= self.time_out_buf
    
    def  _get_phase(self):
        '''计算步态周期的相位。
        '''
        cycle_time = self.cfg.rewards.cycle_time
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_gait_phase(self):
        '''计算步态相位，返回一个掩码，表示每只脚是支撑相还是摆动相。
        '''
        # return float mask 1 is stance, 0 is swing
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        # Add double support phase
        stance_mask = torch.zeros((self.num_envs, 2), device=self.device)
        # left foot stance
        stance_mask[:, 0] = sin_pos >= 0
        # right foot stance
        stance_mask[:, 1] = sin_pos < 0
        # Double support phase
        stance_mask[torch.abs(sin_pos) < 0.1] = 1

        return stance_mask
    

    def compute_ref_state(self):
        '''计算参考状态，包括参考关节位置和动作。
        '''
        phase = self._get_phase()
        sin_pos = torch.sin(2 * torch.pi * phase)
        sin_pos_l = sin_pos.clone()
        sin_pos_r = sin_pos.clone()
        self.ref_dof_pos = torch.zeros_like(self.dof_pos)
        scale_1 = self.cfg.rewards.target_joint_pos_scale
        scale_2 = 2 * scale_1
        # left foot stance phase set to default joint pos
        sin_pos_l[sin_pos_l > 0] = 0
        self.ref_dof_pos[:, 2] = sin_pos_l * scale_1
        self.ref_dof_pos[:, 3] = sin_pos_l * scale_2
        self.ref_dof_pos[:, 4] = sin_pos_l * scale_1
        # right foot stance phase set to default joint pos
        sin_pos_r[sin_pos_r < 0] = 0
        self.ref_dof_pos[:, 8] = sin_pos_r * scale_1
        self.ref_dof_pos[:, 9] = sin_pos_r * scale_2
        self.ref_dof_pos[:, 10] = sin_pos_r * scale_1
        # Double support phase
        self.ref_dof_pos[torch.abs(sin_pos) < 0.1] = 0

        self.ref_action = 2 * self.ref_dof_pos


    def create_sim(self):
        """ Creates simulation, terrain and evironments
            创建仿真、地形和环境。根据配置文件中的地形类型，选择不同的地形创建方法。
        """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = HumanoidTerrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'heightfield':
            self._create_heightfield()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        
        设置用于缩放添加到观察值的噪声的向量。
        """
        noise_vec = torch.zeros(
            self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_vec[0: 5] = 0.  # commands
        noise_vec[5: 17] = noise_scales.dof_pos * self.obs_scales.dof_pos
        noise_vec[17: 29] = noise_scales.dof_vel * self.obs_scales.dof_vel
        noise_vec[29: 41] = 0.  # previous actions
        noise_vec[41: 44] = noise_scales.ang_vel * self.obs_scales.ang_vel   # ang vel
        noise_vec[44: 47] = noise_scales.quat * self.obs_scales.quat         # euler x,y
        return noise_vec


    def step(self, actions):
        '''执行给定动作的仿真步进。如果配置文件中设置了使用参考动作，则将参考动作添加到给定动作中。还包括动态随机化]
        '''
        if self.cfg.env.use_ref_actions:
            actions += self.ref_action
        # dynamic randomization
        delay = torch.rand((self.num_envs, 1), device=self.device)
        actions = (1 - delay) * actions + delay * self.actions
        actions += self.cfg.domain_rand.dynamic_randomization * torch.randn_like(actions) * actions
        return super().step(actions)


    def compute_observations(self):
        '''计算观察值，包括步态相位、参考状态、命令输入、关节位置和速度、基础线速度和角速度等。
        '''
        # print("feet_indices", self.feet_indices)
        # print(self.root_states[:, 2][0])
        phase = self._get_phase()
        self.compute_ref_state()

        sin_pos = torch.sin(2 * torch.pi * phase).unsqueeze(1)
        cos_pos = torch.cos(2 * torch.pi * phase).unsqueeze(1)

        stance_mask = self._get_gait_phase()
        contact_mask = self.contact_forces[:, self.feet_indices, 2] > 5.

        self.command_input = torch.cat(
            (sin_pos, cos_pos, self.commands[:, :3] * self.commands_scale), dim=1)
        
        q = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dq = self.dof_vel * self.obs_scales.dof_vel
        
        diff = self.dof_pos - self.ref_dof_pos

        self.privileged_obs_buf = torch.cat((
            self.command_input,  # 2 + 3
            (self.dof_pos - self.default_joint_pd_target) * \
            self.obs_scales.dof_pos,  # 12
            self.dof_vel * self.obs_scales.dof_vel,  # 12
            self.actions,  # 12
            diff,  # 12
            self.base_lin_vel * self.obs_scales.lin_vel,  # 3
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
            self.rand_push_force[:, :2],  # 3
            self.rand_push_torque,  # 3
            self.env_frictions,  # 1
            self.body_mass / 30.,  # 1
            stance_mask,  # 2
            contact_mask,  # 2
        ), dim=-1)

        obs_buf = torch.cat((
            self.command_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
            q,    # 12D
            dq,  # 12D
            self.actions,   # 12D
            self.base_ang_vel * self.obs_scales.ang_vel,  # 3
            self.base_euler_xyz * self.obs_scales.quat,  # 3
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        if self.add_noise:  
            obs_now = obs_buf.clone() + torch.randn_like(obs_buf) * self.noise_scale_vec * self.cfg.noise.noise_level
        else:
            obs_now = obs_buf.clone()
        self.obs_history.append(obs_now)
        self.critic_history.append(self.privileged_obs_buf)


        obs_buf_all = torch.stack([self.obs_history[i]
                                   for i in range(self.obs_history.maxlen)], dim=1)  # N,T,K

        self.obs_buf = obs_buf_all.reshape(self.num_envs, -1)  # N, T*K
        self.privileged_obs_buf = torch.cat([self.critic_history[i] for i in range(self.cfg.env.c_frame_stack)], dim=1)

    def reset_idx(self, env_ids):
        '''重置指定环境的状态，并清空观察历史和评论历史
        '''
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] *= 0
        for i in range(self.critic_history.maxlen):
            self.critic_history[i][env_ids] *= 0

# ================================================ Rewards ================================================== #
    # reference motion tracking
    def _reward_joint_pos(self):
        """
        Calculates the reward based on the difference between the current joint positions and the target joint positions.
        根据当前关节位置和目标关节位置之间的差异计算奖励。旨在鼓励机器人将关节位置尽量保持在目标位置附近。
        """
        # 获取当前关节位置的副本
        joint_pos = self.dof_pos.clone() # 使用 clone 方法创建一个副本，以避免对原始数据进行修改
        pos_target = self.ref_dof_pos.clone() # 同样使用 clone 方法创建一个副本。
        diff = joint_pos - pos_target # 当前关节位置和目标关节位置之间的差异。
        r = torch.exp(-2 * torch.norm(diff, dim=1)) - 0.2 * torch.norm(diff, dim=1).clamp(0, 0.5)
        '''
        torch.norm(diff, dim=1) 计算每个环境中所有关节位置差异的范数（即欧几里得距离）。
        torch.exp(-2 * torch.norm(diff, dim=1)) 计算一个基于关节位置差异的指数衰减奖励。差异越小，奖励越高。
        torch.norm(diff, dim=1).clamp(0, 0.5) 将关节位置差异限制在 [0, 0.5] 范围内，并乘以 0.2，作为惩罚项。差异越大，惩罚越大。
        最终奖励 r 是指数衰减奖励减去惩罚项。
        '''
        return r

    def _reward_feet_contact_number(self):
        """
        Calculates a reward based on the number of feet contacts aligning with the gait phase. 
        Rewards or penalizes depending on whether the foot contact matches the expected gait phase.
        根据脚接触数量与步态相位的匹配情况计算奖励
        """
        # 获取脚的接触情况，接触力大于5的视为接触
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        '''
        self.contact_forces 是一个张量，包含了所有接触力信息。
        self.feet_indices 是脚的索引，用于从 self.contact_forces 中提取脚的接触力。
        [:, self.feet_indices, 2] 提取脚在 z 轴方向上的接触力。
        > 5. 判断接触力是否大于 5，返回一个布尔张量，表示脚是否接触地面。
        '''
        # 获取步态相位掩码，1表示支撑相，0表示摆动相
        stance_mask = self._get_gait_phase() # 调用 _get_gait_phase 方法获取步态相位掩码,
                                             #返回一个布尔张量，表示每只脚是支撑相（1）还是摆动相（0）。
        
        # 根据接触情况和步态相位掩码计算奖励或惩罚
        # 如果接触情况与步态相位匹配，则奖励1，否则惩罚-0.3
        reward = torch.where(contact == stance_mask, 1, -0.3)
        '''
        torch.where 函数根据条件选择值。
        如果 contact 和 stance_mask 匹配（即接触情况与步态相位一致），则奖励 1。
        否则，惩罚 -0.3。
        '''
        # 返回平均奖励
        return torch.mean(reward, dim=1) # 计算每个环境的平均奖励,返回一个包含每个环境奖励的张量。


    def _reward_feet_clearance(self):
        """
        Calculates reward based on the clearance of the swing leg from the ground during movement.
        Encourages appropriate lift of the feet during the swing phase of the gait.
        根据摆动腿离地高度计算奖励，鼓励在摆动相期间适当抬起脚。
        """
        # 计算脚的接触掩码，接触力大于5的视为接触
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.

        # 获取脚的z坐标并计算z坐标的变化
        feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
        delta_z = feet_z - self.last_feet_z
        self.feet_height += delta_z
        self.last_feet_z = feet_z

        # 计算摆动掩码，1表示摆动相，0表示支撑相
        swing_mask = 1 - self._get_gait_phase()

        # 脚的高度应接近目标高度
        rew_pos = torch.abs(self.feet_height - self.cfg.rewards.target_feet_height) < 0.01 # 判断脚的高度是否接近目标高度。
        rew_pos = torch.sum(rew_pos * swing_mask, dim=1) # 计算摆动相期间脚的高度奖励。
        
        # 重置脚的高度，只有在接触时才重置
        self.feet_height *= ~contact # 在非接触状态下重置脚的高度。
        
        # 返回奖励
        return rew_pos


    # gait
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        根据脚之间的距离计算奖励，惩罚脚靠得太近或太远。
        """
        # 获取脚的位置，只考虑 x 和 y 坐标
        foot_pos = self.rigid_state[:, self.feet_indices, :2] # 提取脚在 x 和 y 轴方向上的位置。
        # 计算脚之间的距离
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1) # 计算距离向量的范数（即欧几里得距离）。
        # 最小和最大允许的脚之间的距离
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        # 计算距离惩罚
        # 如果脚之间的距离小于最小距离，则计算负偏差
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.) # 将差异限制在 [-0.5, 0] 范围内，表示负偏差。
        # 如果脚之间的距离大于最大距离，则计算正偏差
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5) # 将差异限制在 [0, 0.5] 范围内，表示正偏差。
        # 返回奖励，使用指数函数计算基于距离偏差的奖励
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2


    def _reward_knee_distance(self):
        """
        计算奖励函数，基于双腿膝盖之间的距离。
        
        该函数旨在鼓励膝盖之间的距离保持在理想范围内，既不太近也不太远。
        距离过近可能会导致碰撞，过远则可能降低稳定性。
        
        返回:
            float: 基于膝盖距离计算的奖励值。
        """
        """
        Calculates the reward based on the distance between the knee of the humanoid.
        根据膝盖之间的距离计算奖励，惩罚膝盖靠得太近或太远。
        """
        # 获取膝盖位置，其中self.rigid_state包含所有刚体的状态，knee_indices是膝盖索引的集合
        foot_pos = self.rigid_state[:, self.knee_indices, :2]
        # 计算左右膝盖之间的距离
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        # 定义最小理想距离和最大理想距离的一半
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist / 2
        # 计算实际距离与最小理想距离和最大理想距离之差
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.)
        d_max = torch.clamp(foot_dist - max_df, 0, 0.5)
        # 根据距离与理想距离的偏差计算奖励，偏差越大，奖励越小
        return (torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)) / 2

    def _reward_foot_slip(self):
        """
        计算足部滑动的奖励函数。
        
        该函数旨在鼓励机器人保持足部与地面的稳定接触，避免滑动。它通过测量足部的速度和接触力来评估滑动情况。
        当足部与地面的接触力超过一定阈值时，认为足部处于接触状态；然后计算足部的速度，并根据接触状态来调整奖励值。
        如果足部在接触地面时有滑动（速度不为零），则会减少奖励；如果足部稳定（速度为零），则增加奖励.
        """
        """
        Calculates the reward for minimizing foot slip. The reward is based on the contact forces 
        and the speed of the feet. A contact threshold is used to determine if the foot is in contact 
        with the ground. The speed of the foot is calculated and scaled by the contact condition.
        根据脚的滑动情况计算奖励，惩罚脚在接触地面时的滑动。
        """
        # 判断足部是否与地面接触
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # 计算足部速度的范数
        foot_speed_norm = torch.norm(self.rigid_state[:, self.feet_indices, 10:12], dim=2)
        # 根据足部速度计算奖励值
        rew = torch.sqrt(foot_speed_norm)
        # 根据接触状态调整奖励值，只有在足部接触地面时才给予奖励
        rew *= contact
        # 返回所有足部奖励的总和
        return torch.sum(rew, dim=1)

    def _reward_feet_air_time(self):
        """
        计算脚在空中的时间奖励。
        
        这个函数的目的是鼓励机器人在行走时采取更长的步幅，从而增加其脚在空中停留的时间。
        奖励的计算基于脚首次接触地面的时间，该时间被限制在一个最大值以内。
        
        Returns:
            torch.Tensor: 表示每个脚在空中时间的奖励总和。
        """
        """
        Calculates the reward for feet air time, promoting longer steps. This is achieved by
        checking the first contact with the ground after being in the air. The air time is
        limited to a maximum value for reward calculation.
        根据脚在空中的时间计算奖励，鼓励更长的步伐。
        """
        # 判断每个脚是否与地面接触
        contact = self.contact_forces[:, self.feet_indices, 2] > 5.
        # 获取当前步态阶段的掩码
        stance_mask = self._get_gait_phase()
        # 更新接触掩码，包括当前接触、当前步态阶段和上一帧的接触状态
        self.contact_filt = torch.logical_or(torch.logical_or(contact, stance_mask), self.last_contacts)
        # 更新上一帧的接触状态
        self.last_contacts = contact
        # 标记每个脚首次接触地面的瞬间
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        # 更新每个脚在空中的时间
        self.feet_air_time += self.dt
        # 计算奖励，限制空气时间在0到0.5秒之间
        air_time = self.feet_air_time.clamp(0, 0.5) * first_contact
        # 重置脚在空中的时间，如果脚当前接触地面
        self.feet_air_time *= ~self.contact_filt
        # 返回每个脚的空气时间奖励的总和
        return air_time.sum(dim=1)

    # contact
    def _reward_feet_contact_forces(self):
        """
        计算脚部接触力的奖励函数。
        
        目的是鼓励机器人保持接触力在合理范围内，避免过大的接触力导致的损伤或不稳定。
        奖励函数通过对接触力的_norm_（范数）与最大允许接触力之差进行计算。
        如果差值超过400，则差值被clip函数限制在400以内，然后对所有脚部的差值求和。
        
        返回:
        - torch.Tensor: 表示每个时间步对应脚部接触力奖励的向量。
        """
        """
        Calculates the reward for keeping contact forces within a specified range. Penalizes
        high contact forces on the feet.
        根据脚的接触力计算奖励，惩罚过高的接触力。
        """
        # 计算脚部接触力的范数与最大允许接触力之差
        difference = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force
        # 将差值超出400的部分限制在400以内
        clipped_difference = difference.clip(0, 400)
        # 对所有脚部的差值求和，得到每个时间步的奖励值
        return torch.sum(clipped_difference, dim=1)
    
    # vel tracking
    def _reward_tracking_lin_vel(self):
        """
        计算跟踪线性速度的奖励。
        
        这个函数用于评估机器人当前的线性速度与目标速度的偏差，并根据这种偏差计算一个奖励值。
        奖励值的大小与速度偏差成反比，偏差越小，奖励越大。
        
        返回:
            torch.Tensor: 一个标量张量，表示线性速度跟踪的奖励。
        """
        """
        Tracks linear velocity commands along the xy axes. 
        Calculates a reward based on how closely the robot's linear velocity matches the commanded values.
        根据机器人线速度与命令值的匹配情况计算奖励。
        """
        # 计算线性速度误差的平方和
        lin_vel_error = torch.sum(torch.square(
            self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 根据误差计算奖励，误差越小，奖励越大
        return torch.exp(-lin_vel_error * self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        """
        计算基于角速度跟踪的奖励。
        
        该函数旨在评估机器人实际角速度与目标角速度之间的偏差，并根据此偏差计算奖励值。
        偏差越小，奖励值越高，激励机器人更好地跟踪目标角速度。
        
        返回:
            torch.Tensor: 基于角速度跟踪误差计算的奖励值。
        """
        """
        Tracks angular velocity commands for yaw rotation.
        Computes a reward based on how closely the robot's angular velocity matches the commanded yaw values.
        根据机器人角速度与命令值的匹配情况计算奖励。
        """   
        # 计算yaw方向（即z轴）的角速度误差
        ang_vel_error = torch.square(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        
        # 根据误差计算奖励，误差越小，奖励越高
        # 这里使用指数函数来平滑奖励分布，使奖励更灵敏地响应误差变化
        return torch.exp(-ang_vel_error * self.cfg.rewards.tracking_sigma)
    
    def _reward_vel_mismatch_exp(self):
        """
        计算速度不匹配的奖励函数。
        
        该函数旨在评估机器人线性速度和角速度的不匹配程度，并据此计算一个奖励值。
        奖励值的大小与速度的不匹配程度成反比，即速度越接近期望值，奖励越高。
        这鼓励机器人尽可能保持稳定的速度，避免快速变化或波动。
        
        返回:
            c_update: 一个张量，表示速度不匹配的奖励值。
        """
        """
        Computes a reward based on the mismatch in the robot's linear and angular velocities. 
        Encourages the robot to maintain a stable velocity by penalizing large deviations.
        根据机器人线速度和角速度的不匹配情况计算奖励，鼓励保持稳定的速度。
        """
        # 计算线性速度的不匹配程度，使用指数函数对较大的速度偏差进行惩罚
        lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
        # 计算角速度的不匹配程度，同样使用指数函数对较大的角速度偏差进行惩罚
        ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)

        # 计算线性速度和角速度不匹配程度的平均值，用于综合评估速度的稳定性
        c_update = (lin_mismatch + ang_mismatch) / 2.

        return c_update

    def _reward_low_speed(self):
        """
        根据机器人实际速度与目标速度的匹配程度计算奖励值。
        奖励值的设计旨在鼓励机器人保持接近目标速度的行驶速度，并且朝向正确的方向。
        
        Returns:
            torch.Tensor: 一个张量，表示每个时间步的奖励值，用于强化学习的计算。
        """
        """
        Rewards or penalizes the robot based on its speed relative to the commanded speed. 
        This function checks if the robot is moving too slow, too fast, or at the desired speed, 
        and if the movement direction matches the command.
        根据机器人速度相对于命令速度的情况计算奖励，检查机器人是否移动得太慢、太快或在期望的速度范围内。
        """
        # 计算机器人实际速度和目标速度的绝对值，以便进行比较
        # Calculate the absolute value of speed and command for comparison
        absolute_speed = torch.abs(self.base_lin_vel[:, 0])
        absolute_command = torch.abs(self.commands[:, 0])

        # 定义速度条件，判断机器人速度是过低、过高还是在理想范围内
        # Define speed criteria for desired range
        speed_too_low = absolute_speed < 0.5 * absolute_command
        speed_too_high = absolute_speed > 1.2 * absolute_command
        speed_desired = ~(speed_too_low | speed_too_high)

        # 检查机器人实际行驶方向与目标方向是否一致
        # Check if the speed and command directions are mismatched
        sign_mismatch = torch.sign(
            self.base_lin_vel[:, 0]) != torch.sign(self.commands[:, 0])

        # 初始化奖励张量，所有元素为0（默认情况下）
        # Initialize reward tensor
        reward = torch.zeros_like(self.base_lin_vel[:, 0])

        # 根据机器人的速度情况和方向情况，为每个时间步设置相应的奖励值
        # Assign rewards based on conditions
        # Speed too low
        reward[speed_too_low] = -1.0
        # Speed too high
        reward[speed_too_high] = 0.
        # Speed within desired range
        reward[speed_desired] = 1.2
        # Sign mismatch has the highest priority
        reward[sign_mismatch] = -2.0

        # 仅在目标速度大于0.1时才计算奖励，避免因目标速度过小导致的奖励计算误差
        return reward * (self.commands[:, 0].abs() > 0.1)
    
    def _reward_track_vel_hard(self):
        """
        计算奖励函数，用于评估机器人跟踪线性和角速度命令的性能。
        此函数通过对线性速度和角速度误差的指数衰减来奖励准确的跟踪行为，
        并对跟踪误差进行惩罚。误差越小，奖励越高；误差越大，奖励越低。

        返回:
            奖励值，表示机器人跟踪速度命令的准确性。
        """
        """
        Calculates a reward for accurately tracking both linear and angular velocity commands.
        Penalizes deviations from specified linear and angular velocity targets.
        根据机器人线速度和角速度的跟踪情况计算奖励，惩罚偏离指定速度目标的情况。
        """
        # 计算线性速度误差，并对其进行指数衰减
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.norm(
            self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1)
        lin_vel_error_exp = torch.exp(-lin_vel_error * 10)

        # 计算角速度误差，并对其进行指数衰减
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(
            self.commands[:, 2] - self.base_ang_vel[:, 2])
        ang_vel_error_exp = torch.exp(-ang_vel_error * 10)

        # 计算线性误差，用于后续的惩罚
        linear_error = 0.2 * (lin_vel_error + ang_vel_error)

        # 计算奖励值，综合考虑线性和角速度的跟踪误差
        return (lin_vel_error_exp + ang_vel_error_exp) / 2. - linear_error

    # base pos
    def _reward_orientation(self):
        """
        计算奖励函数，用于评估机器人基座定向的奖励。
        
        此函数旨在鼓励机器人保持其基座朝向的稳定性。它通过比较基座的欧拉角和投影的重力向量来评估定向偏差。
        偏离期望定向将受到惩罚，从而激励机器人维持其基座朝向的稳定。
        
        返回:
            float: 基于基座定向稳定性的奖励值。该值是一个在0到1之间的浮点数，
                   其中1表示最大的定向稳定性，0表示最差的稳定性。
        """
        # 计算基座欧拉角的差异，并应用一个指数衰减，以惩罚较大的角度偏差
        quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
        
        # 计算投影的重力向量的长度，并应用一个指数衰减，以惩罚重力向量的较大偏差
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        
        # 平均两个惩罚因素的结果，并将其归一化到0到1之间，作为奖励值返回
        return (quat_mismatch + orientation) / 2.

    def _reward_default_joint_pos(self):
        """
        计算奖励函数，用于评估关节位置与默认目标位置的接近程度。
        重点惩罚yaw（偏航）和roll（滚转）方向的偏差，以鼓励机器人保持其默认姿态。
        
        返回:
        float: 奖励值，基于关节位置偏差的惩罚计算得出。值越接近1，表示关节位置越接近默认目标位置。
        """
        """
        Calculates the reward for keeping joint positions close to default positions, with a focus 
        on penalizing deviation in yaw and roll directions. Excludes yaw and roll from the main penalty.
        根据关节位置与默认位置的偏差计算奖励，重点惩罚偏离偏航和滚转方向的情况。
        """
        # 计算当前关节位置与默认目标位置的偏差
        joint_diff = self.dof_pos - self.default_joint_pd_target
        
        # 提取左和右yaw和roll方向的偏差
        left_yaw_roll = joint_diff[:, :2]
        right_yaw_roll = joint_diff[:, 6: 8]
        
        # 计算左和右yaw和roll方向偏差的范数之和
        yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
        
        # 将yaw和roll方向的偏差限制在一定范围内，并对其进行惩罚
        yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)
        
        # 根据yaw和roll方向的偏差以及关节位置的整体偏差计算奖励值
        return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)

    def _reward_base_height(self):
        """
        计算基于机器人基座高度的奖励函数。
        此函数旨在鼓励机器人的基座高度保持在目标高度附近。
        奖励值会随着基座高度与目标高度的偏差增加而减少。
        
        返回:
        基于基座高度与目标高度偏差的奖励值，为一个张量。
        """
        """
        Calculates the reward based on the robot's base height. Penalizes deviation from a target base height.
        The reward is computed based on the height difference between the robot's base and the average height 
        of its feet when they are in contact with the ground.
        根据机器人基础高度计算奖励，惩罚偏离目标基础高度的情况。
        """
        # 获取当前步态阶段的掩码，用于确定哪些脚正在接触地面
        stance_mask = self._get_gait_phase()
        
        # 计算所有接触地面的脚的平均高度
        measured_heights = torch.sum(
            self.rigid_state[:, self.feet_indices, 2] * stance_mask, dim=1) / torch.sum(stance_mask, dim=1)
        
        # 计算机器人基座的高度，将其调整到与脚的平均高度相比较的水平
        base_height = self.root_states[:, 2] - (measured_heights - 0.05)
        
        # 根据基座高度与目标高度的偏差计算奖励值，奖励值随偏差增加而指数减少
        return torch.exp(-torch.abs(base_height - self.cfg.rewards.base_height_target) * 100)

    def _reward_base_acc(self):
        """
        计算基于基座加速度的奖励。
        
        本函数旨在奖励机器人基座平稳运动的行为。通过惩罚高加速度的变化，鼓励机器人采取更加平滑的运动策略。
        这有助于在机器人学习过程中强调平稳性和运动的连续性。
        
        返回:
            torch.Tensor: 一个标量张量，表示基于基座加速度的奖励值。
        """
        """
        Computes the reward based on the base's acceleration. Penalizes high accelerations of the robot's base,
        encouraging smoother motion.
        根据机器人基础加速度计算奖励，惩罚高加速度的情况，鼓励平滑运动。
        """
        # 计算基座的加速度
        root_acc = self.last_root_vel - self.root_states[:, 7:13]
        
        # 根据加速度的范数计算奖励，范数越大，惩罚越大，因此奖励越小
        rew = torch.exp(-torch.norm(root_acc, dim=1) * 3)
        
        return rew
    
    # energy
    def _reward_torques(self):
        """
        计算扭矩奖励函数。
        
        该函数旨在鼓励机器人使用较低的扭矩来完成任务，因为高扭矩通常意味着更高的能量消耗和潜在的机械磨损。通过
        对关节扭矩的平方和进行惩罚，可以激励机器人寻找更高效、更省力的运动方式。
        
        返回:
            torch.Tensor: 一个标量张量，表示当前扭矩配置的奖励值。该值越小，表示使用的扭矩越低，奖励越高。
        """
        """
        Penalizes the use of high torques in the robot's joints. Encourages efficient movement by minimizing
        the necessary force exerted by the motors.
        根据机器人关节使用的扭矩计算奖励，惩罚高扭矩的情况，鼓励高效运动。
        """
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        """
        计算关节速度的奖励函数。
        
        该函数旨在鼓励机器人的运动尽可能平稳，通过惩罚关节速度的平方和来实现。
        平稳的运动对于许多机器人任务来说都是重要的，因为它可以减少能量消耗，
        避免过快的运动导致的碰撞或失稳。
        
        返回:
            torch.Tensor: 一个标量张量，表示关节速度的惩罚值。这个值越小，
            表明关节速度越平稳，机器人将得到更高的奖励。
        """
        """
        Penalizes high velocities at the degrees of freedom (DOF) of the robot. This encourages smoother and 
        more controlled movements.
        根据机器人关节的速度计算奖励，惩罚高速度的情况，鼓励平滑和受控的运动。
        """
        # 计算所有关节速度的平方和，用于惩罚高速度
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """
        计算关节加速度的奖励函数。
        
        该函数通过比较当前关节速度与上一时刻关节速度的变化量来估计关节的加速度，并对高加速度情况进行惩罚。
        这有助于鼓励机器人执行平稳的运动，从而减少运动过程中可能造成的冲击和磨损。
        
        返回:
            加速度惩罚值的张量，形状为(批量大小, )。
        """
        """
        Penalizes high accelerations at the robot's degrees of freedom (DOF). This is important for ensuring
        smooth and stable motion, reducing wear on the robot's mechanical parts.
        根据机器人关节的加速度计算奖励，惩罚高加速度的情况，确保平滑和稳定的运动，减少机械部件的磨损。
        """
        # 计算关节加速度的估计值
        acceleration_estimate = (self.last_dof_vel - self.dof_vel) / self.dt
        # 计算加速度的平方和，用于后续的惩罚计算
        acceleration_squared = torch.square(acceleration_estimate)
        # 对所有关节的加速度平方和进行求和，得到一个针对所有样本的加速度惩罚值
        return torch.sum(acceleration_squared, dim=1)
    def _reward_collision(self):
        """
        计算碰撞惩罚。

        该函数通过检测机器人与环境碰撞的力度来计算惩罚值。具体来说，它关注的是机器人特定部位（由penalised_contact_indices定义）所受的接触力是否超过了一定阈值（0.1）。如果超过这个阈值，说明发生了较为严重的碰撞，会给予较高的惩罚。

        返回:
            torch.Tensor: 一个标量张量，表示由于碰撞而产生的惩罚值之和。
        """
        """
        Penalizes collisions of the robot with the environment, specifically focusing on selected body parts.
        This encourages the robot to avoid undesired contact with objects or surfaces.
        根据机器人与环境的碰撞情况计算奖励，惩罚不希望的接触，鼓励避免碰撞。
        """
        # 计算所有被惩罚的碰撞接触点的力的模长。
        collision_forces = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        # 筛选出碰撞力超过0.1的点。
        collision_mask = collision_forces > 0.1
        # 对满足碰撞条件的点进行求和，得到总的碰撞惩罚值。
        collision_reward = torch.sum(collision_mask, dim=1)
        return collision_reward
    def _reward_action_smoothness(self):
        """
        计算动作平滑度奖励。
        
        该函数通过惩罚连续动作之间的过大差异来鼓励机器人的动作平滑性。这有助于减少机械应力，
        并促进流畅的运动。具体来说，它计算了三个项：当前动作与上一动作的差异、当前动作与前前动作的差异，
        以及当前动作的绝对值之和。这些项的组合有助于平衡动作的平滑度和幅度。
        
        返回:
            torch.Tensor: 表示动作平滑度奖励的张量。
        """
        """
        Encourages smoothness in the robot's actions by penalizing large differences between consecutive actions.
        This is important for achieving fluid motion and reducing mechanical stress.
        根据连续动作之间的差异计算奖励，惩罚大幅度的动作变化，鼓励平滑的动作，减少机械应力。
        """
        # 计算当前动作与上一动作之间的差异，并对差异进行平方求和。
        term_1 = torch.sum(torch.square(
            self.last_actions - self.actions), dim=1)
        
        # 计算当前动作与前前动作的差异，并考虑与上一动作的差异，对差异进行平方求和。
        term_2 = torch.sum(torch.square(
            self.actions + self.last_last_actions - 2 * self.last_actions), dim=1)
        
        # 添加一个较小的项，惩罚动作的绝对值之和，以鼓励动作在一定程度上的平滑。
        term_3 = 0.05 * torch.sum(torch.abs(self.actions), dim=1)
        
        # 返回三个项的和作为动作平滑度的奖励。
        return term_1 + term_2 + term_3


'''
### 奖励设置概述

在这个代码中，机器人强化学习的奖励设置通过多个函数来实现，每个函数计算一个特定方面的奖励或惩罚。奖励设置的目的是引导机器人在训练过程中学习到更好的运动策略，具体包括以下几个方面：

1. **参考运动跟踪**
2. **步态和脚部行为**
3. **速度跟踪**
4. **基础位置和姿态**
5. **能量消耗**
6. **碰撞和动作平滑性**

### 详细奖励函数

#### 1. 参考运动跟踪

- **_reward_joint_pos**: 根据当前关节位置和目标关节位置之间的差异计算奖励，鼓励机器人将关节位置尽量保持在目标位置附近。

#### 2. 步态和脚部行为

- **_reward_feet_contact_number**: 根据脚接触数量与步态相位的匹配情况计算奖励，鼓励机器人在正确的步态相位下接触地面。
- **_reward_feet_clearance**: 根据摆动腿离地高度计算奖励，鼓励在摆动相期间适当抬起脚。
- **_reward_feet_distance**: 根据脚之间的距离计算奖励，惩罚脚靠得太近或太远。
- **_reward_knee_distance**: 根据膝盖之间的距离计算奖励，惩罚膝盖靠得太近或太远。
- **_reward_foot_slip**: 根据脚的滑动情况计算奖励，惩罚脚在接触地面时的滑动。
- **_reward_feet_air_time**: 根据脚在空中的时间计算奖励，鼓励更长的步伐。
- **_reward_feet_contact_forces**: 根据脚的接触力计算奖励，惩罚过高的接触力。

#### 3. 速度跟踪

- **_reward_tracking_lin_vel**: 根据机器人线速度与命令值的匹配情况计算奖励。
- **_reward_tracking_ang_vel**: 根据机器人角速度与命令值的匹配情况计算奖励。
- **_reward_vel_mismatch_exp**: 根据机器人线速度和角速度的不匹配情况计算奖励，鼓励保持稳定的速度。
- **_reward_low_speed**: 根据机器人速度相对于命令速度的情况计算奖励，检查机器人是否移动得太慢、太快或在期望的速度范围内。
- **_reward_track_vel_hard**: 根据机器人线速度和角速度的跟踪情况计算奖励，惩罚偏离指定速度目标的情况。

#### 4. 基础位置和姿态

- **_reward_orientation**: 根据机器人基础姿态的平稳性计算奖励，惩罚偏离期望姿态的情况。
- **_reward_default_joint_pos**: 根据关节位置与默认位置的偏差计算奖励，重点惩罚偏离偏航和滚转方向的情况。
- **_reward_base_height**: 根据机器人基础高度计算奖励，惩罚偏离目标基础高度的情况。
- **_reward_base_acc**: 根据机器人基础加速度计算奖励，惩罚高加速度的情况，鼓励平滑运动。

#### 5. 能量消耗

- **_reward_torques**: 根据机器人关节使用的扭矩计算奖励，惩罚高扭矩的情况，鼓励高效运动。
- **_reward_dof_vel**: 根据机器人关节的速度计算奖励，惩罚高速度的情况，鼓励平滑和受控的运动。
- **_reward_dof_acc**: 根据机器人关节的加速度计算奖励，惩罚高加速度的情况，确保平滑和稳定的运动，减少机械部件的磨损。

#### 6. 碰撞和动作平滑性

- **_reward_collision**: 根据机器人与环境的碰撞情况计算奖励，惩罚不希望的接触，鼓励避免碰撞。
- **_reward_action_smoothness**: 根据连续动作之间的差异计算奖励，惩罚大幅度的动作变化，鼓励平滑的动作，减少机械应力。

### 总结

这些奖励函数的设计目的是通过多方面的评估和反馈，引导机器人在训练过程中逐步优化其运动策略。具体来说，这些奖励函数鼓励机器人：

- 精确跟踪参考运动
- 保持合适的步态和脚部行为
- 准确跟踪速度命令
- 维持稳定的基础位置和姿态
- 最小化能量消耗
- 避免碰撞并保持动作平滑

通过这些奖励函数，机器人可以在训练过程中不断调整和改进其行为，最终实现更高效、更稳定的运动。
'''