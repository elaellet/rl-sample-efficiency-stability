import argparse
import glob
import os

import numpy as np

from .utils import (calculate_auc,
                    get_n_runs,
                    parse_run_name,
                    print_banner,
                    smooth_curve)
from .visualizations import plot_learning_curves

def print_descriptive_statistics(results, n_eval_episodes=200):
    returns = results['returns']
    successes = results['successes']

    mean_reward = np.mean(returns)
    std_reward = np.std(returns)
    min_reward = np.min(returns)
    max_reward = np.max(returns)
    success_rate = (successes / n_eval_episodes) * 100

    print(f'├── Mean Reward:              {mean_reward:.3f}')
    print(f'├── Std Reward:               {std_reward:.3f}')
    print(f'├── Min Reward:               {min_reward:.3f}')
    print(f'├── Max Reward:               {max_reward:.3f}')
    print(f'├── Success Count:            {successes}/{n_eval_episodes}')
    print(f'├── Success Rate:             {success_rate:.2f}%')

if __name__ == '__main__':
    ENV_IDS = [
        'PandaReachJoints-v3', 'PandaPushJoints-v3', 
        'PandaSlideJoints-v3', 'PandaPickAndPlaceJoints-v3',
        'PandaStackJoints-v3', 'PandaFlipJoints-v3',
        'PandaReachJointsDense-v3', 'PandaPushJointsDense-v3', 
        'PandaSlideJointsDense-v3', 'PandaPickAndPlaceJointsDense-v3',
        'PandaStackJointsDense-v3', 'PandaFlipJointsDense-v3'
    ]
    ALGOS = ['heuristic', 'reinforce', 'sac']

    parser = argparse.ArgumentParser(description='Analyze a policy in a Gymnasium environment.')
    parser.add_argument('--env-id', help='Gymnasium environment ID', required=True, choices=ENV_IDS)
    parser.add_argument('--algo', help='Algorithm of the model to load', default='sac', type=str, choices=ALGOS)
    parser.add_argument('--log-folder',  help='Path to the log folder', default='./logs', type=str)
    parser.add_argument('--plot-filename', help='Filename of learning curves plot', default='learning_curves', type=str)

    args = parser.parse_args()

    print_banner('Individual Run Analysis')

    pathname_pattern = os.path.join(f'{args.log_folder}_*', 'evaluations.npz')
    
    npz_files = glob.glob(pathname_pattern)
    npz_files.sort(key=get_n_runs)

    best_success_rates = list()
    auc_scores = list()
    converged_timesteps = list()

    for idx, file in enumerate(npz_files):
        try:
            run = np.load(file)
            successes = run['successes']
            timesteps = run['timesteps']

            if successes.size > 0:
                mean_success_rate = np.mean(successes, axis=1)
                
                best_idx = np.argmax(mean_success_rate)
                best_success_rate = mean_success_rate[best_idx]
                best_timestep = timesteps[best_idx]
                best_success_rates.append(best_success_rate)

                auc = calculate_auc(timesteps, mean_success_rate)
                auc_scores.append(auc)
                
                # Using smoothed curve to handle oscillation.
                smoothed = smooth_curve(mean_success_rate, 5)
                threshold_indices = np.where(smoothed >= 0.90)[0]
                
                run_name = os.path.basename(os.path.dirname(file))
                algo, reward, seed = parse_run_name(run_name)
                
                print(f'[ {algo} | {reward} | {seed} ]')
                print(f'├── Best Success:      {best_success_rate * 100:.2f}% (at timestep {best_timestep:,})')
                print(f'├── Norm. AUC:         {auc:.3f}')
                
                if len(threshold_indices) > 0:
                    solved_idx_smooth = threshold_indices[0]
                    shift = (len(timesteps) - len(smoothed))
                    solved_idx_raw = solved_idx_smooth + shift
                    converged_timesteps.append(timesteps[solved_idx_raw])
                    
                    print(f'└── Time-to-Threshold: {timesteps[solved_idx_raw]:,} timesteps')
                else:
                    print(f'└── Time-to-Threshold: Not reached')
                print('-'*60)
        except Exception as e:
            print(f'└── An unexpected error occurred, reading {file}: {e}')

    if best_success_rates:
        mean_best = np.mean(best_success_rates)
        std_best = np.std(best_success_rates)

        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)

        total_runs = len(best_success_rates)
        converged_runs = len(converged_timesteps)

        print_banner('Overall Experiment Summary', newline_top=True)
        print(f'├── Total Runs:        {total_runs}')
        print(f'├── Converged Runs:    {converged_runs}/{total_runs} ({converged_runs / total_runs * 100}%)')
        if converged_runs > 0:
            mean_timesteps = int(np.mean(converged_timesteps))
            std_timesteps = int(np.std(converged_timesteps))

            print(f'├── Mean Timesteps:    {mean_timesteps:,} ± {std_timesteps:,} timesteps')
        else:
            print(f'├── Mean Timesteps:    N/A (No converged runs)')
        print(f'├── Mean Best Success: {mean_best * 100:.2f}% ± {std_best * 100:.2f}%')
        print(f'└── Mean Norm. AUC:    {mean_auc:.2f} ± {std_auc:.2f}')
    else:
        print(f'└── No valid run found.')

    plot_learning_curves(pathname_pattern,
                         algo,
                         args.env_id,
                         plot_filename=args.plot_filename)
