import argparse
import os
import re
from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import panda_gym
import tensorflow as tf
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import set_random_seed

from .analyze import print_descriptive_statistics
from .policies import heuristic_policy, GaussianPolicyModel
from .visualizations import plot_stability_boxplot
from .utils import (find_best_model_and_stats,
                    make_env,
                    print_banner,
                    print_separator,
                    unvectorize_observation)

def evaluate(vec_env,
             policy,
             n_timesteps,
             n_eval_episodes=200,
             batch_mode=False,
             speed=3.0):
    '''
    Runs a performance evaluation loop for a given policy.

    This function runs a specified number of episodes (`n_eval_episodes`) to
    measure the performance of a policy without training. It correctly
    handles different policy types:
    1.  Stable Baselines3 (BaseAlgorithm): Expects vectorized, normalized observations.
    2.  TensorFlow (tf.keras.Model): Expects single, mixed (norm/raw) observations.
    3.  Heuristic (callable): Expects single, raw observations.

    Parameters:
        speed (float): The speed multiplier, passed to the heuristic policy.

    Returns:
        dict: A dictionary containing a list of `returns` (total reward
            per episode) and the total `successes` count.
    '''
    robot_id = vec_env.envs[0].unwrapped.robot.sim._bodies_idx['panda']
    env_id = vec_env.envs[0].spec.id

    returns = list()
    successes = 0
    
    for episode in range(n_eval_episodes):
        episode_rewards = 0
        vec_env.seed(episode)
        vec_obs = vec_env.reset()

        for _ in range(n_timesteps):
            # SAC
            if isinstance(policy, BaseAlgorithm):
                vec_action, _ = policy.predict(vec_obs, deterministic=True)
            # REINFORCE
            elif isinstance(policy, tf.keras.Model):
                norm_obs = unvectorize_observation(vec_obs)

                obs_vec = norm_obs['observation']
                gripper_state = obs_vec[:6]
                block_state = obs_vec[6:]
                desired_state = norm_obs['desired_goal']

                model_input = {
                    'gripper': gripper_state[np.newaxis],
                    'block': block_state[np.newaxis],
                    'goal': desired_state[np.newaxis]
                }

                means, _ = policy(model_input, training=False)
                action = tf.clip_by_value(means[0], -1.0, 1.0).numpy()
                vec_action = np.array([action])
            # Heuristic Policy
            else:
                raw_obs = unvectorize_observation(vec_env.get_original_obs())
                action = policy(raw_obs, env_id, robot_id, speed)
                vec_action = np.array([action])

            vec_obs, rewards, dones, infos = vec_env.step(vec_action)
            
            if not batch_mode:
                vec_env.render()
                #time.sleep(0.03)

            episode_rewards += rewards[0]

            if dones[0]:
                if infos[0].get('is_success'):
                    successes += 1
                break

        returns.append(episode_rewards)

    return {'returns': returns, 'successes': successes}

def run_batch_evaluation(args):
    '''
    Scans folder, groups experiments, evaluates seeds, and aggregates stats.
    '''
    print_banner(f'BATCH EVALUATION ({args.algo.upper()})')

    # Regex expects format: 'experiment_name_seed' (e.g., sac_dense_42 -> sac_dense)
    groups = defaultdict(list)
    subfolders = [folder.path for folder in os.scandir(args.batch_folder) if folder.is_dir()]
    
    for folder in subfolders:
        folder_name = os.path.basename(folder)
        match = re.match(r'(.+)_(\d+)$', folder_name)
        if match:
            exp_id = match.group(1)
            groups[exp_id].append(folder)
        else:
            groups[folder_name].append(folder)

    if not groups:
        print(f'├── No valid subfolders found in {args.batch_folder}')
        return
    
    eval_stats = dict()

    for exp_id, folders in groups.items():
        print_separator(f'Experiment Config')
        
        success_rates = list()
        valid_seeds = list()

        curr_env_id = args.env_id
        
        # If folder is 'sac_dense' but CLI env is 'Sparse', switch to 'Dense'.
        if 'dense' in exp_id.lower() and 'sparse' in curr_env_id.lower():
            if '-v' in curr_env_id:
                curr_env_id = curr_env_id.replace('-v', 'Dense-v')
            else:
                curr_env_id += 'Dense'
        # If folder is 'sparse' but CLI env is 'Dense', switch to 'Sparse'.
        elif 'sparse' in exp_id.lower() and 'dense' in curr_env_id.lower():
            curr_env_id = curr_env_id.replace('Dense', '')
        
        print(f'├── Auto-switched Env to {curr_env_id}')

        for folder in sorted(folders):
            seed_name = os.path.basename(folder)
            print(f'├── Run ID:                   {seed_name}')
            
            try:
                model_path, stats_path = find_best_model_and_stats(folder)
                
                if not model_path:
                    print(f'│   └── No valid model found.')
                    continue

                if not stats_path:
                    print(f'│   └── No valid stats found.')
                    continue

                model_path = model_path.replace('\\', '/')
                stats_path = stats_path.replace('\\', '/')

                vec_env = make_env(curr_env_id, 
                                   1, 
                                   args.seed, 
                                   stats_path=stats_path)
                print(f'├── Model Path:               {model_path}')
                
                policy = SAC.load(model_path, env=vec_env)
                
                eval_results = evaluate(vec_env, 
                                        policy,
                                        args.n_timesteps, 
                                        args.n_eval_episodes, 
                                        True,
                                        args.speed)
                vec_env.close()

                success_rate = (eval_results['successes'] / args.n_eval_episodes)
                success_rates.append(success_rate)
                valid_seeds.append(seed_name)
                
                print(f'│   └── Success Rate:         {success_rate * 100:.2f}%')
            except Exception as e:
                print(f'│   └── Evaluation Failed: {e}')

        if success_rates:
            eval_stats[exp_id] = success_rates
            mean_success_rate = np.mean(success_rates) * 100
            std_success_rate = np.std(success_rates) * 100
            print(f'├── Config Summmary')
            print(f'│   ├── Mean Success Rate:    {mean_success_rate:.2f}% ± {std_success_rate:.2f}%')
            print(f'│   └── Valid Runs:           {len(valid_seeds)}/{len(folders)}')
        else:
            print(f'└── No valid runs found.')
    
    if eval_stats:
        print_separator('Stability Box Plot')
        plot_stability_boxplot(eval_stats)

def main():
    ENV_IDS = [
        'PandaReachJoints-v3', 'PandaPushJoints-v3', 
        'PandaSlideJoints-v3', 'PandaPickAndPlaceJoints-v3',
        'PandaStackJoints-v3', 'PandaFlipJoints-v3',
        'PandaReachJointsDense-v3', 'PandaPushJointsDense-v3', 
        'PandaSlideJointsDense-v3', 'PandaPickAndPlaceJointsDense-v3',
        'PandaStackJointsDense-v3', 'PandaFlipJointsDense-v3'
    ]
    ALGOS = ['heuristic', 'reinforce', 'sac']

    parser = argparse.ArgumentParser(description='Evaluate a policy in a Gymnasium environment.')
    parser.add_argument('--env-id', help='Gymnasium environment ID', type=str, choices=ENV_IDS)
    parser.add_argument('--algo', help='Algorithm of the model to load (Required if using --model-path)', default='heuristic', type=str, choices=ALGOS)
    parser.add_argument('--seed', help='Random seed (Default: 42)', default=42, type=int)    

    folder_group = parser.add_mutually_exclusive_group()
    folder_group.add_argument('--log-folder', help='Path to a single log folder', type=str)
    folder_group.add_argument('--batch-folder', help='Path to the parent folder containing multiple logs', type=str)

    parser.add_argument('--n-eval-episodes', help='Number of episodes to run for evaluation (Default: 200)', default=200, type=int)
    parser.add_argument('--n-timesteps', help='Number of timesteps per episode (Default: 50)', default=50, type=int)
    parser.add_argument('--speed', help='Speed multiplier for heuristic policy (Default: 3.0)', default=3.0, type=float)

    vec_env = None

    try:
        args = parser.parse_args()

        if args.batch_folder:
            if args.algo == 'heuristic':
                print('[ERROR] Batch mode is for trained agents, not heuristics.')
                return
            run_batch_evaluation(args)
            return

        set_random_seed(args.seed)

        policy_eval = None
        policy_name = args.algo.upper()

        print_banner(f'Evaluating {policy_name} on {args.env_id}', newline_top=True)

        if args.algo != 'heuristic':
            model_path, stats_path = find_best_model_and_stats(args.log_folder)
            print(f'├── Auto-detected best model in {args.log_folder}')

        model_path = model_path.replace('\\', '/')
        stats_path = stats_path.replace('\\', '/')

        vec_env = make_env(args.env_id, 
                           1,
                           args.seed,
                           is_eval=True,
                           stats_path=stats_path)

        if args.algo == 'heuristic':
            policy_eval = heuristic_policy
        elif args.algo == 'reinforce':
            model_input = {
                'gripper': tf.ones((1, 6)),
                'block': tf.ones((1, 12)),
                'goal': tf.ones((1, 3))
            }

            policy_eval = GaussianPolicyModel()
            policy_eval(model_input)
            policy_eval.load_weights(model_path)
            stats_path = args.stats_path
        elif args.algo == 'sac':
            policy_eval = SAC.load(model_path, env=vec_env)
        else:
            raise ValueError(f'└── Unknown algorithm:          {args.algo}')

        if args.algo != 'heuristic':
            print(f'├── Model Path:               {model_path}')
        print(f'├── Seed:                     {args.seed}')
        print(f'├── Episodes:                 {args.n_eval_episodes}')
        print(f'├── Timesteps per Episode:    {args.n_timesteps}')

        eval_results = evaluate(vec_env, 
                                policy_eval, 
                                args.n_timesteps,
                                args.n_eval_episodes,
                                False,
                                args.speed)
        
        print_descriptive_statistics(eval_results, args.n_eval_episodes)
    except KeyboardInterrupt:
        print(f'{"└── " if args.batch_folder else "├── "}Evaluation interrupted by user (Ctrl+C)')
    except Exception as e:
        print(f'├── An unexpected error occurred: {e}')
        import traceback
        traceback.print_exc()        
    finally:
            if not args.batch_folder:
                if vec_env is not None:
                    print('└── Cleaning up and closing environment...')
                    vec_env.close()
                else:
                    print('└── No environment to save or close.')

                print_banner(f'{policy_name} Evaluation Complete')
            else:
                print_banner(f'Batch Evaluation ({args.algo.upper()}) Complete')

if __name__ == '__main__':
    main()