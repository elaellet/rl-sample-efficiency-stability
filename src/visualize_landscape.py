import argparse

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import panda_gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecCheckNan

from .utils import find_best_model_and_stats, make_env
from .visualizations import plot_value_landscape

def get_q_values(model, env, distances):
    '''
    Generates artificial observations, normalizes them, and queries the Critic.
    '''
    q_values = list()
    
    obj_pos = np.array([0.0, 0.0, 0.05])
    goal_pos = np.array([0.3, 0.3, 0.05]) 
    
    print(f'Sampling {len(distances)} points for {env}...')

    for dist in distances:
        gripper_pos = obj_pos + np.array([dist, 0.0, 0.05])
        
        # [gripper_pos(3), gripper_vel(3),
        #  obj_pos(3), obj_rot(3), 
        #  obj_vel(3), obj_ang_vel(3)]
        obs = np.concatenate([
            gripper_pos,
            [0, 0, 0],         
            obj_pos,
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        
        raw_obs_space = {
            'achieved_goal': obj_pos.copy(),
            'desired_goal': goal_pos.copy(),
            'observation': obs.copy()
        }
        
        if isinstance(env, VecCheckNan):
            norm_obs_space = {
                'achieved_goal': env.venv._normalize_obs(raw_obs_space['achieved_goal'], env.venv.obs_rms['achieved_goal']), #_normalize_key('achieved_goal', raw_obs_space, env),
                'desired_goal': env.venv._normalize_obs(raw_obs_space['desired_goal'], env.venv.obs_rms['desired_goal']), #_normalize_key('desired_goal', raw_obs_space, env),
                'observation': env.venv._normalize_obs(raw_obs_space['observation'], env.venv.obs_rms['observation'])  #_normalize_key('observation', raw_obs_space, env),
            }
        else:
            norm_obs_space = raw_obs_space

        obs_tensor = model.policy.obs_to_tensor(norm_obs_space)[0]
        
        action = np.zeros((1, env.action_space.shape[0]))
        action_tensor = torch.as_tensor(action, device=model.device).float()
        
        with torch.no_grad():
            q1, q2 = model.critic(obs_tensor, action_tensor)
            q_val = torch.min(q1, q2).item()
            
        q_values.append(q_val)

    return q_values

def main():
    ENV_IDS = [
        'PandaReachJoints-v3', 'PandaPushJoints-v3', 
        'PandaSlideJoints-v3', 'PandaPickAndPlaceJoints-v3',
        'PandaStackJoints-v3', 'PandaFlipJoints-v3',
        'PandaReachJointsDense-v3', 'PandaPushJointsDense-v3', 
        'PandaSlideJointsDense-v3', 'PandaPickAndPlaceJointsDense-v3',
        'PandaStackJointsDense-v3', 'PandaFlipJointsDense-v3'
    ]

    parser = argparse.ArgumentParser()

    parser.add_argument('--env-id', help='Gymnasium environment ID', type=str, choices=ENV_IDS)
    parser.add_argument('--seed', help='Random seed (Default: 42)', default=42, type=int)    
    parser.add_argument('--dense-folder', help='Path to log folder of dense reward', type=str, required=True)
    parser.add_argument('--her-folder', help='Path to log folder of sparse reward', type=str, required=True)

    args = parser.parse_args()

    dense_env_id = args.env_id if 'Dense' in args.env_id else args.env_id + 'Dense'
    dense_env_id = dense_env_id.replace('-v3Dense', 'Dense-v3')
    sparse_env_id = args.env_id.replace('Dense', '')

    d_model_path, d_stats_path = find_best_model_and_stats(args.dense_folder)
    h_model_path, h_stats_path = find_best_model_and_stats(args.her_folder)
    
    d_env = make_env(dense_env_id, 1, args.seed, stats_path=d_stats_path)
    d_model = SAC.load(d_model_path, env=d_env)
    
    h_env = make_env(sparse_env_id, 1, args.seed, stats_path=h_stats_path)
    h_model = SAC.load(h_model_path, env=h_env)

    distances = np.linspace(0.0, 0.3, 50) 
    
    d_q_values = get_q_values(d_model, d_env, distances)
    h_q_values = get_q_values(h_model, h_env, distances)
    
    plot_value_landscape(distances,
                         d_q_values,
                         h_q_values,
                         args.seed)

if __name__ == '__main__':
    main()