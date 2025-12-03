import glob
import os
import re

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecCheckNan

def calculate_auc(timesteps, values):
    '''
    Calculates the normalized Area Under the Curve (AUC).
    '''
    if len(timesteps) < 2:
        return 0.0
    
    # Trapezoidal integration
    area = np.trapezoid(values, timesteps)
    
    # Normalize by total time duration
    total_time = timesteps[-1] - timesteps[0]
    if total_time == 0:
        return 0.0
        
    return area / total_time

def calculate_distance(pos1, pos2):
    '''
    Calculates the Euclidean distance between two points.
    '''
    return np.linalg.norm(pos1 - pos2)

def find_best_model_and_stats(log_folder):
    '''
    Finds the best model (based on evaluations.npz) and 
    the closest VecNormalize stats.
    '''
    npz_file = os.path.join(log_folder, 'evaluations.npz')

    if not os.path.exists(npz_file):
        raise FileNotFoundError(f'[ERROR] No evaluations.npz file found in {log_folder}')

    run = np.load(npz_file)
    timesteps = run['timesteps']
    mean_success = np.mean(run['successes'], axis=1)

    best_idx = np.argmax(mean_success)
    best_timestep = timesteps[best_idx]

    pkl_files = glob.glob(os.path.join(log_folder, '*.pkl'))

    if not pkl_files:
        raise FileNotFoundError(f'[ERROR] No *.pkl files found in {log_folder}')
    
    candidates = []
    for file in pkl_files:
        match = re.search(r'vecnormalize_(\d+)_steps\.pkl', file)
        if match:
            step = int(match.group(1))
            candidates.append((step, file))
    
    if not candidates:
        raise ValueError('[ERROR] Could not parse timestep integers from .pkl filenames.')

    best_match = min(candidates, key=lambda x: abs(x[0] - best_timestep))
    found_timestep, found_stats_path = best_match
    diff = abs(found_timestep - best_timestep)

    best_model_path = os.path.join(log_folder, 'best_model.zip')
    checkpoint_path = os.path.join(log_folder, f'rl_model_{found_timestep}_steps.zip')

    model_path = None
    if os.path.exists(best_model_path):
        model_path = best_model_path
    elif os.path.exists(checkpoint_path):
        print(f'├── [INFO] No best_model.zip file found. Using checkpoint at {found_timestep} steps')
        model_path = checkpoint_path

    if not model_path:
        raise FileNotFoundError(f'[ERROR] No valid model zip found near timestep {best_timestep}')

    print(f'├── Best Model Timestep:      {best_timestep:,}')
    print(f'├── Closest Statistics Found: {found_timestep:,} (Difference: {diff:,} timesteps)')
    
    return model_path, found_stats_path

def get_n_runs(file):
    parent_folder = os.path.dirname(file)
    match = re.search(r'_(\d+)$', parent_folder)

    return int(match.group(1)) if match else 0

def make_env(env_id, 
             n_envs, 
             seed, 
             algo_dir=None, 
             is_eval=False, 
             stats_path=None):
    '''
    Creates, vectorizes, normalizes, and checks a Gymnasium environment.

    This function can either create a new normalized environment or
    load the statistics from a saved VecNormalize environment.

    If `stats_path` is provided and exists:
    1.  Loads the VecNormalize stats from the path.
    2.  Applies them to a new environment.
    3.  Sets the env to evaluation mode (`training=False`).

    If `stats_path` is None:
    1.  Creates a new environment from scratch.
    2.  Wraps it in a new VecNormalize.

    Parameters:
        algo_dir (str, optional): Path to the algorithm's log directory.
            Used for saving monitor logs. Defaults to None.
        stats_path (str, optional): Path to a saved VecNormalize
            statistics file (`.pkl`). If provided, stats are loaded.
            If None, a new normalization wrapper is created.
            Defaults to None.

    Returns:
        VecCheckNan: The fully wrapped and configured vectorized environment.
    '''
    render_mode = 'human' if is_eval else 'rgb_array'
    env_kwargs = {'render_mode': render_mode}
    norm_obs_keys = ['observation', 'achieved_goal', 'desired_goal']
    if algo_dir:
        monitor_path = os.path.join(algo_dir, 'monitor/')
    else:
        monitor_path = None

    vec_env = make_vec_env(env_id,
                           n_envs=n_envs,
                           seed=seed,
                           env_kwargs=env_kwargs,
                           monitor_dir=monitor_path)
    
    stats_path = stats_path.replace('\\', '/')
    
    if stats_path and os.path.exists(stats_path):
        print(f'├── Stats Path:               {stats_path}')
        vec_env = VecNormalize.load(stats_path, vec_env)
        # No update at test time.
        vec_env.training = False
        # No reward normalization needed at test time.
        vec_env.norm_reward = False
    else:
        if stats_path and not os.path.exists(stats_path):
            print(f'├── Warning:              {stats_path} provided but not found. Creating new environment.')

        vec_env = VecNormalize(vec_env,
                               norm_obs=True, 
                               norm_reward=True,
                               clip_obs=10.0, 
                               norm_obs_keys=norm_obs_keys)
    
    vec_env = VecCheckNan(vec_env, raise_exception=True)
    
    return vec_env

def parse_run_name(folder_name):
    parts = folder_name.split('_')
    
    if len(parts) >= 4:
        algo = parts[0].upper()
        if len(parts) == 5:
            algo = algo + '+' + parts[1].upper()
            reward = parts[2].capitalize()
        else:
            reward = parts[1].capitalize()
        seed = parts[-1]
        
        return algo, reward, seed
    return folder_name, 'N/A', 'N/A'

def plot_environment(env, figsize=(5, 4)):
    plt.figure(figsize=figsize)
    img = env.render()
    plt.imshow(img)
    plt.axis('off')

    return img

def print_banner(title, 
                 width=60, 
                 char='=', 
                 newline_top=False, 
                 newline_bottom=False):
    if newline_top: print()    

    total_width = max(width, len(title) + 4)
    separator = char * total_width

    content_space = total_width - 2
    centered_title = f' {title} '.center(content_space)
    
    print(f'{separator}')
    print(f'{char}{centered_title}{char}')
    print(f'{separator}')

    if newline_bottom: print()

def print_separator(title, 
                    width=60,
                    char='-', 
                    newline_top=False, 
                    newline_bottom=False):
    if newline_top: print()

    print(f' {title} '.center(width, char))

    if newline_bottom: print()

def smooth_curve(values, window_size=5):
    '''
    Applies a rolling average smoothing to the data using valid convolution.

    This function computes the moving average of the input values to reduces noise.
    It uses 'valid' convolution, meaning no padding is added at the edges.

    1. Creates a window of weights (1/window_size).
    2. Convolves this window over the input values.
    3. Returns a shorter array (length = input_length - window_size + 1).

    Parameters:
        window_size (int, optional): The size of the moving window.
            Defaults to 5.
    '''
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size) / window_size, mode='valid')

def unvectorize_observation(vec_obs, idx=0):
    '''
    Extracts a single observation dictionary from 
    a vectorized observation.

    A VecEnv returns observations as a dictionary where each key
    maps to a batch of observations (e.g., `obs['observation']`
    is an array of shape (`n_envs, 18`)). This helper extracts
    the observation for a single environment.

    Parameters:
        idx (int): The environment index to extract.

    Returns:
        dict: A single, non-vectorized observation dictionary.
    '''
    return {key: vec_obs[key][idx] for key in vec_obs}