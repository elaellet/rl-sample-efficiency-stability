import glob
import os

import numpy as np
import matplotlib.pyplot as plt

from .utils import (calculate_auc, 
                    get_n_runs, 
                    print_separator, 
                    smooth_curve)

def plot_learning_curves(pathname_pattern,
                         algo,
                         env_id, 
                         metric='successes',
                         save_path='./outputs',
                         plot_filename='learning_curves.png',
                         window_size=5):
    '''
    Finds all 'evaluations.npz' files for a given task/algo,
    calculates the specified metric, and plots them on one graph.
    '''
    print_separator('Learning Curves Plot', newline_top=True)

    pathname_pattern = pathname_pattern.replace('\\', '/')

    print(f'├── Searching for logs with pattern: {pathname_pattern}')

    npz_files = glob.glob(pathname_pattern)
    
    if not npz_files:
        print(f'└── [WARNING] No \'evaluations.npz\' files found.')
        return

    plt.figure(figsize=(12, 7))
    print(f'├── Found {len(npz_files)} evaluation files. Plotting...')
    
    npz_files.sort(key=get_n_runs)

    for file in npz_files:
        try:
            n_run = get_n_runs(file)
            label = f'Seed {n_run}'

            run = np.load(file)
            timesteps = run['timesteps']
            
            if metric == 'successes':
                vals = run['successes']
                ylabel = 'Success Rate'
            elif metric == 'results':
                vals = run['results']
                ylabel = 'Mean Reward'
            else:
                vals = run.get(metric, run['results'])
                ylabel = metric.replace('_', ' ').title()
            
            if len(vals.shape) > 1:
                mean_vals = np.mean(vals, axis=1)
            else:
                mean_vals = vals
                
            smoothed_vals = smooth_curve(mean_vals, window_size)

            cutoff = len(timesteps) - len(smoothed_vals)
            start_idx = cutoff // 2
            smoothed_timesteps = timesteps[start_idx : start_idx + len(smoothed_vals)]

            p = plt.plot(smoothed_timesteps,
                         smoothed_vals,
                         linestyle='-',
                         linewidth=2,
                         alpha=1.0,
                         label=label,
                         zorder=2)          
            line_color = p[0].get_color()

            plt.plot(timesteps, 
                     mean_vals, 
                     linestyle='-', 
                     linewidth=1,
                     alpha=0.4,
                     color=line_color,
                     zorder=1)
                  
            auc = calculate_auc(timesteps, mean_vals)
            p[0].set_label(f'{label} (AUC: {auc:.2f})')
                     
        except Exception as e:
            print(f'└── [ERROR] Processing file {file}: {e}')

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, plot_filename)
    full_path = full_path.replace('\\', '/')

    plt.title(f'{algo.upper()} {ylabel} vs. Timesteps ({env_id})')
    plt.xlabel('Timesteps')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(full_path)
    plt.close()

    print(f'└── Plot saved to {full_path}')

def plot_stability_boxplot(stats, 
                           save_path='./outputs',
                           plot_filename='policy_stability_analysis.png'):
    '''
    Generates a box plot comparing success rates across experiments.
    data: dict { 'experiment_name': [rate1, rate2, ...] }
    '''
    labels = sorted(stats.keys())
    values = [stats[label] for label in labels]

    plt.figure(figsize=(10, 6))
    
    p = plt.boxplot(values,
                    tick_labels=labels,
                    patch_artist=True,
                    medianprops=dict(color='black'))

    colors = ['lightcoral', 'lightblue', 'lightgreen']
    for idx, patch in enumerate(p['boxes']):
        patch.set_facecolor(colors[idx % len(colors)])

    for idx, run in enumerate(values):
        x = [idx + 1] * len(run)
        plt.scatter(x, run, color='red', alpha=0.6, zorder=3, label='Individual Seed' if idx == 0 else '')

    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.8, label='Success Threshold (90%)')

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, plot_filename)
    full_path = full_path.replace('\\', '/')

    plt.title(f'Policy Success Distribution (N=5)')
    plt.ylabel('Success Rate (0.0 - 1.0)')
    plt.ylim(-0.05, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.savefig(full_path, dpi=300)
    plt.close()

    print(f'└── Plot saved to {full_path}')

def plot_value_landscape(distances,
                         d_q_values,
                         h_q_values,
                         seed,
                         save_path='./outputs'):
    '''
    Generates a dual-axis plot comparing Dense vs HER Critic values across distance.

    Parameters:
        distances (np.array): Array of distance values (in meters).
        d_q_values (list): Critic values for the SAC (Dense) agent.
        h_q_values (list): Critic values for the SAC+HER (Sparse) agent.
    '''
    _, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Distance from Gripper to Object (cm)', fontsize=12)
    ax1.set_ylabel('SAC (Dense) Value', color=color, fontsize=12)
    l1, = ax1.plot(distances * 100, d_q_values, color=color, linewidth=3, marker='o', markevery=5, label='SAC (Dense Reward)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('SAC+HER (Sparse) Value', color=color, fontsize=12)
    l2, = ax2.plot(distances * 100, h_q_values, color=color, linewidth=3, marker='s', markevery=5, label='SAC + HER (Sparse Reward)')
    ax2.tick_params(axis='y', labelcolor=color)

    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, f'value_landscape_{seed}.png')
    full_path = full_path.replace('\\', '/')

    plt.title(f'Value Function Landscape (Seed {seed})', fontsize=14)
    lines = [l1, l2]
    plt.legend(lines, [l.get_label() for l in lines], loc='upper center', fontsize=12)
    plt.savefig(full_path, dpi=300)
    plt.close()
    
    print(f'└── Plot saved to {full_path}')