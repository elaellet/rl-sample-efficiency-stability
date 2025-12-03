# import argparse
# import os
# import time

# import gymnasium as gym
# import numpy as np
# import panda_gym
# import tensorflow as tf
# from stable_baselines3 import HerReplayBuffer, SAC
# from stable_baselines3.common.callbacks import (BaseCallback,
#                                                 CallbackList,
#                                                 CheckpointCallback, 
#                                                 EvalCallback, 
#                                                 StopTrainingOnRewardThreshold)
# from stable_baselines3.common.utils import set_random_seed

# from .policies import GaussianPolicyModel
# from .utils import make_env, print_separator, unvectorize_observation

# def _play_one_step(vec_env, vec_obs, model):
#     '''
#     Plays one step in the environment using the policy model.

#     This function prepares the input for the model by concatenating the
#     `observation` and `desired_goal` from the `obs` dictionary.
#     It then performs a forward pass within a GradientTape to get the
#     action distribution parameters (means, stds). An action is
#     stochastically sampled from this distribution, and the loss is
#     calculated as the negative log-probability of that action.
#     Finally, it computes the gradients, clips the action to valid
#     bounds, and executes the step in the environment.

#     Parameters:
#         vec_env (VecEnv): The NaN-checked, normalized, vectorized Gymnasium environment.
#         vec_obs (dict): The current vectorized observation dictionary from the environment.
#         model (tf.keras.Model): The (GaussianPolicyModel) to use for action selection.

#     Returns:
#         tuple: A tuple containing:
#         vec_obs (dict): The vectorized observation dictionary.
#         reward (float): The reward received from the step.
#         terminated (bool): True if the episode terminated.
#         truncated (bool): True if the episode was truncated.
#         grads (list): A list of gradient tensors for the model's variables.
#     '''
#     norm_obs = unvectorize_observation(vec_obs)
#     obs_vec = norm_obs['observation']
    
#     # Split the 18-dim observation vector
#     # [0:6] is gripper state (6 dims)
#     # [6:18] is block state (12 dims)
#     gripper_state = obs_vec[:6]
#     block_state = obs_vec[6:]
#     desired_state = norm_obs['desired_goal']

#     model_input = {
#         'gripper': gripper_state[np.newaxis],
#         'block': block_state[np.newaxis],
#         'goal': desired_state[np.newaxis]
#     }

#     with tf.GradientTape() as tape:
#         # Expect a batch, containing a single instance.
#         means, stds = model(model_input)
#         # Sample a 7D action from 7 Gaussian distributions.
#         action = tf.random.normal(shape=tf.shape(means), mean=means, stddev=stds)
#         # Calculate the negative log-probability of the action.
#         # Loss for a Gaussian whose gradient is required.
#         neg_log_prob = 0.5 * tf.square((action - means) / stds) + tf.math.log(stds) + 0.5 * tf.math.log(2 * np.pi)
#         # Sum the loss for all 7 actions.
#         loss = tf.reduce_sum(neg_log_prob)

#     grads = tape.gradient(loss, model.trainable_variables)
#     # Clip the action to fit the environment's [-1, 1] bounds.
#     clipped_action = tf.clip_by_value(action[0], -1.0, 1.0).numpy()

#     vec_obs, rewards, dones, _ = vec_env.step(np.array([clipped_action]))

#     return vec_obs, rewards[0], dones[0], grads

# def _play_multiple_episodes(vec_env, 
#                             n_eps, 
#                             n_max_steps, 
#                             model):
#     ''' 
#     Collects experience by playing multiple episodes with the model.

#     This function iterates for a specified number of episodes (`n_eps`).
#     In each episode, it repeatedly calls `play_one_step` to interact
#     with the environment. It collects the `reward` and `grads`
#     from every step until the episode ends (`done`) or 
#     the `n_max_steps` limit is reached.

#     Parameters:
#         vec_env (VecEnv): The NaN-checked, normalized, vectorized Gymnasium environment.
#         model (tf.keras.Model): The (GaussianPolicyModel) to use for action selection.

#     Returns:
#         tuple: A tuple containing:
#             all_rewards (list): A list of lists, where each inner list
#                 contains the rewards for one episode.
#             all_grads (list): A list of lists, where each inner list
#                 contains the gradients for each step of one episode.
#     '''
#     all_rewards = list()
#     all_grads = list()

#     for _ in range(n_eps):
#         curr_rewards = list()
#         curr_grads = list()

#         vec_obs = vec_env.reset()

#         for _ in range(n_max_steps):
#             vec_obs, reward, done, grads = _play_one_step(vec_env, vec_obs, model)

#             curr_rewards.append(reward)
#             curr_grads.append(grads)

#             if done:
#                 break
                
#         all_rewards.append(curr_rewards)
#         all_grads.append(curr_grads)

#     return all_rewards, all_grads

# # return = rewards[0] + rewards[1] * γ + rewards[2] * γ^2 + ...
# def _discount_rewards(rewards, gamma):
#     '''
#     Computes the discounted sum of future rewards for each step in an episode.

#     This function iterates backward through a list of rewards for a single
#     episode. It calculates the reward-to-go at each timestep, where the
#     value of a step is its immediate reward plus the discounted value
#     of the next step.

#     Returns:
#         np.ndarray: An array containing the discounted sum of future
#             rewards for each step.
#     '''
#     discounted = np.array(rewards)
#     for step in range(len(rewards) - 2, -1, -1):
#         discounted[step] += discounted[step + 1] * gamma

#     return discounted

# def _discount_and_normalize_rewards(all_rewards, gamma):
#     '''
#     Calculates and normalizes the discounted rewards over all episodes.

#     This function first computes the discounted rewards for every
#     episode using the `discount_rewards` function. It then flattens all
#     these values into a single array to calculate a global mean and
#     standard deviation. Finally, it normalizes all discounted
#     rewards using this mean and std.

#     Parameters:
#         all_rewards (list): A list of lists, where each inner list
#             contains the rewards for one episode.

#     Returns:
#         list: A list of np.ndarrays, where each array contains the
#             normalized discounted rewards for the corresponding episode.
#     '''
#     all_discounted_rewards = [_discount_rewards(rewards, gamma) for rewards in all_rewards]
#     flat_rewards = np.concatenate(all_discounted_rewards)

#     mean_reward = flat_rewards.mean()
#     std_reward = flat_rewards.std()

#     return [(discounted_rewards - mean_reward) / (std_reward)
#             for discounted_rewards in all_discounted_rewards]

# def train_one_iteration(vec_env, 
#                         model, 
#                         optimizer, 
#                         n_eps, 
#                         n_max_steps, 
#                         gamma):
#     '''
#     Runs a single training iteration for the policy gradient model.

#     This function collects experience by calling `play_multiple_episodes`.
#     It then computes the normalized advantage for each action by calling
#     `discount_and_normalize_rewards`. Finally, it calculates the
#     weighted mean of the gradients for all trainable variables and
#     applies them to the optimizer to update the model.

#     Parameters:
#         vec_env (VecEnv): The NaN-checked, normalized, vectorized Gymnasium environment.
#         model (tf.keras.Model): The policy model to be trained.
#         optimizer (tf.keras.optimizers.Optimizer): The optimizer for the gradient update.

#     Returns:
#         float: The `total_rewards` collected across all episodes in this iteration.
#     '''
#     # Collect experience from multiple episodes.
#     all_rewards, all_grads = _play_multiple_episodes(
#         vec_env, n_eps, n_max_steps, model
#     )

#     # Calculate total rewards for logging.
#     total_rewards = sum(map(sum, all_rewards))

#     # Compute each action's normalized advantage.
#     all_final_rewards = _discount_and_normalize_rewards(all_rewards, gamma)

#     # Calculate the weighted mean of the gradients for trainable variables over all episodes and all steps.
#     all_mean_grads = []
#     for var_idx in range(len(model.trainable_variables)):
#         mean_grads = tf.reduce_mean(
#             [
#                 final_reward * all_grads[ep_idx][step][var_idx]
#                 for ep_idx, final_rewards in enumerate(all_final_rewards)
#                 for step, final_reward in enumerate(final_rewards)
#             ],
#             axis=0,
#         )
#         all_mean_grads.append(mean_grads)

#     # Apply the mean gradients using the optimizer to the model.
#     # The model's trainable variables will be tweaked.
#     optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

#     return total_rewards

# def train_reinforce(train_env, eval_env, args):
#     '''
#     The main REINFORCE training loop.
    
#     (eval_env is not used, but included to match the function signature).

#     Parameters:
#         train_env (VecNormalize): The vectorized, normalized training environment.
#         eval_env (VecNormalize): The vectorized, normalized evaluation environment.
#             (Not used by REINFORCE, but matches the function signature).
#         args (argparse.Namespace): An object containing all command-line
#             arguments.

#     Returns:
#         tf.keras.Model: The trained policy model.
#     '''
#     learning_rate = args.learning_rate
#     gamma = args.gamma
#     n_iters = args.n_iters
#     n_eps_iter = args.n_eps_iter
#     n_max_steps = args.n_max_steps
#     chkpt_freq = args.chkpt_freq

#     CHKPT_SAVE_PATH = args.CHKPT_SAVE_PATH

#     print(f'├── Learning Rate:          {learning_rate}')
#     print(f'├── Discount Factor:        {gamma}')
#     print(f'├── Iterations:             {n_iters}')
#     print(f'├── Episodes per Iteration: {n_eps_iter}')
#     print(f'├── Max Steps per Episode:  {n_max_steps}')
#     print(f'├── Checkpoint Frequency:   {chkpt_freq}')

#     tf.random.set_seed(42)
    
#     ph_gripper = tf.ones((1, 6)) 
#     ph_block = tf.ones((1, 12))
#     ph_goal = tf.ones((1, 3))
        
#     ph_input = {
#         'gripper': ph_gripper,
#         'block': ph_block,
#         'goal': ph_goal
#     }

#     model = GaussianPolicyModel()
#     # Keras infers all the input and output shapes for every layer and create the actual weights.
#     # The model is built
#     model(ph_input)
#     optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

#     print('├── Model Initialized:')
#     model.summary()

#     start_time = time.time()

#     for iter in range(n_iters):
#         total_rewards = train_one_iteration(train_env, 
#                                             model, 
#                                             optimizer, 
#                                             n_eps_iter, 
#                                             n_max_steps, 
#                                             gamma)
        
#         print(f'\r├── Iteration: {iter + 1}/{n_iters} | Mean Reward: {total_rewards / n_eps_iter:.1f}', end='')

#         if (iter + 1) % chkpt_freq == 0:
#             print(f'\n├── [Checkpoint] Saving model and env stats at iteration {iter+1}...')

#             model_save_path = os.path.join(CHKPT_SAVE_PATH, f'reinforce_{iter+1}_iterations.weights.h5')
#             model.save_weights(model_save_path)
            
#             stats_save_path = os.path.join(CHKPT_SAVE_PATH, f'vec_normalize_{iter+1}_iterations.pkl')
#             train_env.save(stats_save_path)

#             print(f'├── [Checkpoint] Saved to {CHKPT_SAVE_PATH}')

#     total_time = time.time() - start_time
#     print(f'\n├── Training loop finished in {total_time:.2f} seconds.')

#     return model

# class SaveVecNormalizeOnBest(BaseCallback):
#     '''
#     Saves the VecNormalize statistics of the training environment
#     when a new best model is found by the EvalCallback.
    
#     save_path: (str) path to save the statistics
#     '''
#     def __init__(self, save_path, verbose=0):
#         super().__init__(verbose)

#         if os.path.isdir(save_path):
#             self.save_path = os.path.join(save_path, 'vec_normalize_best.pkl')
#         else:
#             self.save_path = save_path

#     def _on_step(self):
#         vec_env = self.model.get_vec_normalize_env()

#         if vec_env is not None:
#             if self.verbose > 0:
#                 print(f'- Saving new best VecNormalize stats to {self.save_path}')
#             vec_env.save(self.save_path)
#         else:
#             if self.verbose > 0:
#                 print(f'- Warning: No VecNormalize environment found, not saving stats.')    
        
#         return True
    
# def train_sac(train_env, eval_env, args):
#     '''
#     The main SAC training loop.
    
#     Uses the provided `train_env` for training and `eval_env` for
#     the `EvalCallback`.

#     Returns:
#         stable_baselines3.sac.SAC: The trained SAC model.
#     '''    
#     n_envs = args.n_envs
#     seed = args.seed
#     learning_rate = args.learning_rate
#     gamma = args.gamma
#     n_timesteps = args.n_timesteps
#     eval_freq = args.eval_freq
#     n_eval_eps = args.n_eval_eps
#     batch_size = args.batch_size
#     buffer_size = args.buffer_size

#     CHKPT_SAVE_PATH = args.CHKPT_SAVE_PATH
#     BEST_MODEL_SAVE_PATH = args.BEST_MODEL_SAVE_PATH
#     TENSORBOARD_LOG_DIR = args.TENSORBOARD_LOG_DIR

#     print(f'- Learning Rate:          {learning_rate}')
#     print(f'- Discount Factor:        {gamma}')
#     print(f'- Total Timesteps:        {n_timesteps}')
#     print(f'- Evaluation Frequency:   {eval_freq}')
#     print(f'- Evaluation per Episode: {n_eval_eps}')
#     print(f'- Batch Size:             {batch_size}')
#     print(f'- Buffer Size:            {buffer_size}')

#     save_freq = n_timesteps // 10 # Or 20
#     chkpt_cb = CheckpointCallback(
#         save_freq=max(save_freq // n_envs, 1),
#         save_path=CHKPT_SAVE_PATH,
#         name_prefix='her_sac',
#         save_replay_buffer=True,
#         save_vecnormalize=True,
#         verbose=1
#     )
#     # Stop training if mean reward > -0.05
#     stop_train_cb = StopTrainingOnRewardThreshold(reward_threshold=-0.05, 
#                                                   verbose=1)
#     save_stats_cb = SaveVecNormalizeOnBest(save_path=BEST_MODEL_SAVE_PATH,
#                                            verbose=1)
#     cb_on_best = CallbackList([stop_train_cb, save_stats_cb])
#     eval_cb = EvalCallback(eval_env,
#                            callback_on_new_best=cb_on_best,
#                            n_eval_episodes=n_eval_eps,
#                            eval_freq=max(eval_freq // n_envs, 1),
#                            log_path=BEST_MODEL_SAVE_PATH,
#                            best_model_save_path=BEST_MODEL_SAVE_PATH,
#                            deterministic=True,
#                            render=False,
#                            verbose=1,
#                            warn=True)
#     cb = CallbackList([chkpt_cb, eval_cb])

#     hps = {
#         'learning_rate': learning_rate, # Can also be a schedule
#         'buffer_size': buffer_size,
#         'batch_size': batch_size,
#         'tau': 0.005,
#         'gamma': gamma,
#         # HER needs a full episode to do its magic. It needs to see the final state to go back and relabel failed attempts. 
#         # By setting train_freq to 1 episode, run one full episode, add all its transitions to the buffer (and let HER work on them), and then start the training updates'
#         'train_freq': (1, 'episode'),
#         # 1 episode, do N gradient steps, where N is the number of steps collected in that episode.
#         'gradient_steps': -1,
#         # n_critics: Key part of SAC (Clipped Double-Q)
#         'policy_kwargs': dict(net_arch=[256, 256], n_critics=2)
#     }

#     model = SAC(
#         'MultiInputPolicy',
#         train_env,
#         replay_buffer_class=HerReplayBuffer,
#         replay_buffer_kwargs=dict(
#             n_sampled_goal=4,
#             goal_selection_strategy='future'
#         ),
#         tensorboard_log=TENSORBOARD_LOG_DIR,
#         verbose=1,
#         seed=seed,
#         **hps
#     )

#     print('- Model Initialized:')
#     print(model.policy)

#     start_time = time.time()

#     model.learn(
#         total_timesteps=n_timesteps,
#         callback=cb,
#         log_interval=100 # Log to console every 100 episodes.
#     )

#     total_time = time.time() - start_time
#     print(f'\n- Training loop finished in {total_time:.2f} seconds.')

#     return model

# def main():
#     '''
#     Parses command-line arguments and dispatches to the correct training function.

#     This is the main entry point for the training script. It sets up the
#     `argparse.ArgumentParser` to accept user inputs, organizing them by
#     category (General, Shared, REINFORCE, SAC).

#     Based on the `--algo` argument, it calls the corresponding training
#     function (e.g., `train_reinforce` or `train_sac`).
#     '''
#     ENV_IDS = [
#         'PandaReachJoints-v3', 'PandaPushJoints-v3', 
#         'PandaSlideJoints-v3', 'PandaPickAndPlaceJoints-v3',
#         'PandaStackJoints-v3', 'PandaFlipJoints-v3'
#         'PandaReachJointsDense-v3', 'PandaPushJointsDense-v3', 
#         'PandaSlideJointsDense-v3', 'PandaPickAndPlaceJointsDense-v3',
#         'PandaStackJointsDense-v3', 'PandaFlipJointsDense-v3'
#     ]

#     ALGOS = ['reinforce', 'sac']

#     parser = argparse.ArgumentParser(description='Train an RL agent.')

#     env_group = parser.add_argument_group('Environment Settings')
#     env_group.add_argument('--env-id', type=str, required=True, choices=ENV_IDS, help='Gymnasium environment ID')
#     env_group.add_argument('--n-envs', type=int, default=1, help='Number of parallel environments (Default: 1)')    
#     env_group.add_argument('--algo', type=str, required=True, choices=ALGOS, help='Algorithm to train: \'reinforce\' or \'sac\'')
    
#     shared_group = parser.add_argument_group('Shared Hyperparameters')
#     shared_group.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for the optimizer(s) (Default: 0.01)')
#     shared_group.add_argument('--gamma', type=float, default=0.95, help='Discount factor (gamma) (Default: 0.95)')

#     reinforce_group = parser.add_argument_group('REINFORCE (On-Policy) Settings')
#     reinforce_group.add_argument('--n-iters', type=int, default=1000, help='Number of training iterations (Default: 1000)')
#     reinforce_group.add_argument('--n-eps-iter', type=int, default=20, help='Number of episodes to collect per training iteration (Default: 20)')
#     reinforce_group.add_argument('--n-max-steps', type=int, default=50, help='The maximum number of steps per episode (Default: 50)')
#     reinforce_group.add_argument('--chkpt-freq', type=int, default=100, help='Frequency to save a checkpoint every N iterations (Default: 100)')

#     sac_group = parser.add_argument_group('SAC (Off-Policy) Settings')
#     sac_group.add_argument('--eval-freq', type=int, default=25_000, help='Frequency to run evaluation (in steps) (Default: 25,000)')
#     sac_group.add_argument('--n-eval-eps', type=int, default=20, help='Number of episodes to run for evaluation (Default: 20)')
#     sac_group.add_argument('--buffer-size', type=int, default=1_000_000, help='Size of the replay buffer (Default: 1,000,000)')
#     sac_group.add_argument('--batch-size', type=int, default=256, help='Size of the mini-batch (Default: 256)')
#     sac_group.add_argument('--n-timesteps', type=int, default=1_000_000, help='Total training steps (Default: 1,000,000)')

#     args = parser.parse_args()

#     SEEDS = [1550809597]
#     best_success_rates = list()
#     eval_paths = list()

#     train_env = None
#     eval_env = None
#     model = None

#     for seed in SEEDS:
#         try:
#             args.seed = seed

#             env_id = args.env_id
#             algo = args.algo
#             if algo == 'sac':
#               args.n_envs = 1
#               print(f'!!! SAC/HER detected. Forcing n_envs = 1 !!!')
#             n_envs = args.n_envs

#             set_random_seed(seed)

#             policy = algo.upper()

#             LOG_DIR = './logs/'

#             print_separator(f'Training {policy} on {env_id} (Seed: {seed})', 80)
            
#             ENV_DIR = os.path.join(LOG_DIR, f'{algo}/')
#             os.makedirs(ENV_DIR, exist_ok=True)

#             if algo == 'reinforce':
#                 ALGO_DIR = os.path.join(ENV_DIR, f'reinforce_{seed}/')
#                 args.MODEL_SAVE_PATH = os.path.join(ALGO_DIR, 'reinforce_final.weights.h5')
#                 args.CHKPT_SAVE_PATH = os.path.join(ALGO_DIR, 'checkpoints/')
#                 args.STATS_SAVE_PATH = os.path.join(ALGO_DIR, 'vec_normalize_final.pkl')

#                 os.makedirs(ALGO_DIR, exist_ok=True)
#                 os.makedirs(args.CHKPT_SAVE_PATH, exist_ok=True)

#                 train_fn = train_reinforce
#             elif algo == 'sac':
#                 ALGO_DIR = os.path.join(ENV_DIR, f'sac_{seed}/')
#                 args.TENSORBOARD_LOG_DIR = os.path.join(ALGO_DIR, 'tensorboard/')
#                 args.MODEL_SAVE_PATH = os.path.join(ALGO_DIR, 'sac_final.zip')
#                 args.BEST_MODEL_SAVE_PATH = os.path.join(ALGO_DIR, 'evaluation/')
#                 args.CHKPT_SAVE_PATH = os.path.join(ALGO_DIR, 'checkpoints/')
#                 args.STATS_SAVE_PATH = os.path.join(ALGO_DIR, 'vec_normalize_final.pkl')

#                 os.makedirs(ALGO_DIR, exist_ok=True)
#                 os.makedirs(args.TENSORBOARD_LOG_DIR, exist_ok=True)
#                 os.makedirs(args.BEST_MODEL_SAVE_PATH, exist_ok=True)
#                 os.makedirs(args.CHKPT_SAVE_PATH, exist_ok=True)

#                 eval_paths.append(args.BEST_MODEL_SAVE_PATH)

#                 train_fn = train_sac
#             else:
#                 raise ValueError(f'Unknown algorithm: {algo}')
            
#             print(f'├── Number of Environments:  {n_envs}')

#             train_env = make_env(env_id,
#                                  n_envs,
#                                  seed, 
#                                  ALGO_DIR)
            
#             # Save the stats from the training env first, so the eval_env can load them.
#             EVAL_STATS_SAVE_PATH = os.path.join(ALGO_DIR, 'vec_normalize_initial.pkl')
#             print(f'├── Saving intial VecNormalize stats to {EVAL_STATS_SAVE_PATH}...')
#             train_env.save(EVAL_STATS_SAVE_PATH)

#             eval_env = make_env(env_id,
#                                 1,
#                                 seed+1,
#                                 ALGO_DIR,
#                                 stats_path=EVAL_STATS_SAVE_PATH)
                    
#             model = train_fn(train_env, eval_env, args)
#         except KeyboardInterrupt:
#             print(f'\n├── Run {seed} interrupted by user (Ctrl+C)')
#             print('├── Saving final model and env stats before exiting...')
#         except Exception as e:
#             print(f'\n├── An unexpected error occurred on run {seed}: {e}')
#             print('├── Saving final model and env stats before crashing...')
#             import traceback
#             traceback.print_exc() # Print the full error stack trace
#         finally:
#             if model is not None:
#                 if algo == 'reinforce':
#                     print(f'├── Saving final REINFORCE model weights to {args.MODEL_SAVE_PATH}')
#                     model.save_weights(args.MODEL_SAVE_PATH)
#                 elif algo == 'sac':
#                     print(f'├── Saving final SAC model to {args.MODEL_SAVE_PATH}')
#                     model.save(args.MODEL_SAVE_PATH)
#             else:
#                 print(f'├── No model to save for seed {seed}.')

#             if train_env is not None:
#                 print(f'├── Saving final VecNormalize stats to {args.STATS_SAVE_PATH}')
#                 train_env.save(args.STATS_SAVE_PATH)

#                 print('├── Cleaning up and closing training environment...')
#                 train_env.close()
#             else:
#                 print(f'├── No training environment to save or close for seed {seed}.')

#             if eval_env is not None:
#                 print('└── Cleaning up and closing evaluation environment...')
#                 eval_env.close()
#             else:
#                 print(f'└── No evaluation environment to close seed {seed}.')

#             print_separator(f'{policy} Training Complete (Seed: {seed})', 80)

# if __name__ == '__main__':
#     main()