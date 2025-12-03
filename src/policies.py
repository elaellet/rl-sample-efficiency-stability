import numpy as np
import pybullet as p
import tensorflow as tf

from .utils import calculate_distance

def _calculate_inverse_velocity_kinematics(robot_id, gripper_vel):
    '''
    Calculates the required joint velocities for a gripper velocity.

    This function uses the PyBullet physics engine to compute the Jacobian
    pseudo-inverse for the robot's end-effector. This solves the inverse
    velocity kinematics problem, translating Cartesian (x, y, z) velocity
    into joint-space velocities.

    Parameters:
        robot_id (int): The unique PyBullet ID for the robot.
        gripper_vel (np.ndarray): A 3D vector for the target 
            (x, y, z) velocity of the gripper.

    Returns:
        np.ndarray: A 7D action vector (7 joint velocities)
            clipped to the [-1, 1] range.
    '''
    n_joints = p.getNumJoints(robot_id)
    
    # Get information/states for all 12 joints.
    all_joint_info = [p.getJointInfo(robot_id, i) for i in range(n_joints)]
    all_joint_states = p.getJointStates(robot_id, range(n_joints))
    # Find the indices of only the movable joints (revolute or prismatic).
    movable_joint_indices = [info[0] for info in all_joint_info if info[2] != p.JOINT_FIXED]
    
    # Create new lists containing data only for the movable joints.
    joint_positions = [all_joint_states[i][0] for i in movable_joint_indices]
    
    # The number of degrees of freedom is the count of movable joints.
    dof = len(movable_joint_indices)

    gripper_link_idx = n_joints - 1
    
    jacobian_linear, _ = p.calculateJacobian(
        robot_id,
        gripper_link_idx,
        [0, 0, 0],
        joint_positions,  # 9 joint positions.
        [0] * dof,    # 9 zero velocities.
        [0] * dof     # 9 zero accelerations.
    )

    # Only control the 7 arm joints.
    jacobian_linear = np.array(jacobian_linear)[:, :7] 
    jacobian_inverse = np.linalg.pinv(jacobian_linear)
    # dq = J^(-1)⋅dx
    joint_velocities = jacobian_inverse @ np.array(gripper_vel)

    # Clip the velocities to be within the environment's action bounds.
    return np.clip(joint_velocities, -1.0, 1.0)

def _reach_policy(obs, robot_id, speed):
    '''
    Heuristic policy logic for the `Reach` task.

    This policy calculates the vector from the gripper's current position
    to the desired goal. It then commands a velocity along this vector,
    solving the task with a simple straight-line approach.

    Returns:
        np.ndarray: The 7D action vector calculated by the kinematics helper.
    '''    
    curr_pos = obs['achieved_goal']
    target_pos = obs['desired_goal']
    gripper_vel = (target_pos - curr_pos) * speed
    action = _calculate_inverse_velocity_kinematics(robot_id, gripper_vel)

    return action

def _push_policy(obs, robot_id, speed):
    '''
    Heuristic policy logic for the `Push` task.

    This function implements a 2-step state-based policy.
    Step 1: Move the gripper to a pre-push position located behind the block,
             aligned with the block-to-target vector.
    Step 2: Once aligned, move the gripper through the block's position
             towards the target, pushing it.

    Returns:
        np.ndarray: The 7D action vector calculated by the kinematics helper.
    '''    
    gripper_pos = obs['observation'][0:3]
    block_pos = obs['achieved_goal']
    target_pos = obs['desired_goal']

    vec_block_to_target = target_pos - block_pos
    # Add epsilon to avoid division by zero if the block is at the target position.
    vec_unit_dir = vec_block_to_target / (calculate_distance(target_pos, block_pos) + 1e-6)
    # 5cm offset.
    off_dist = 0.05
    pre_push_pos = block_pos - (vec_unit_dir * off_dist)

    dist_to_pre_push = calculate_distance(gripper_pos, pre_push_pos)

    if dist_to_pre_push > 0.01:
        # Move to the pre-push position.
        gripper_vel = (pre_push_pos - gripper_pos) * speed
    else:
        # Push the block to the target position.
        gripper_vel = (target_pos - gripper_pos) * speed

    action = _calculate_inverse_velocity_kinematics(robot_id, gripper_vel)

    return action

def heuristic_policy(obs, 
                     env_id, 
                     robot_id, 
                     speed=3.0):
    '''
    Master heuristic policy that dispatches to task-specific logic.

    This function acts as a router. It inspects the `env_id` string to
    determine which task is active (e.g., `reach`, `push`) and then calls
    the appropriate internal policy function.

    Returns:
        np.ndarray: The 7D action vector from the selected internal policy.
    
    Raises:
        ValueError: If no policy logic is found for the given `env_id`.
    '''
    env_id_lower = env_id.lower()

    if 'reach' in env_id_lower:
        return _reach_policy(obs, robot_id, speed)
    elif 'push' in env_id_lower:
        return _push_policy(obs, robot_id, speed)
    else:
        raise ValueError(f'No heuristic policy available for task: {env_id}')

class GaussianPolicyModel(tf.keras.Model):
    '''
    A custom Keras model that defines a continuous Gaussian policy
    using multiple, structured inputs.

    This model processes the gripper state, block state, and desired goal
    through separate input layers. This allows the network to learn
    specialized features for each component before combining them
    in shared hidden layers.

    Architecture:
    1. (Input) Gripper State (6,) -> Dense(32) -> Gripper Features
    2. (Input) Block State (12,)  -> Dense(32) -> Block Features
    3. (Input) Goal State (3,)    -> Dense(32) -> Goal Features
    4. Concatenate [Grip_Feat, Block_Feat, Goal_Feat] -> Combined (96,)
    5. Combined -> Dense(64) -> Dense(64)
    6. Output -> mean_layer(7), std_layer(7)
    '''
    def __init__(self):
        super().__init__()
        
        # Specialized Input Encoders.
        self.gripper_encoder = tf.keras.layers.Dense(32, activation='relu')
        self.block_encoder = tf.keras.layers.Dense(32, activation='relu')
        self.goal_encoder = tf.keras.layers.Dense(32, activation='relu')

        # Shared Hidden Layers (after combination)
        self.shared_hidden_1 = tf.keras.layers.Dense(64, activation='relu')
        self.shared_hidden_2 = tf.keras.layers.Dense(64, activation='relu')
        
        # An output head for the 7 mean values
        self.mean_layer = tf.keras.layers.Dense(7)
        # An output head for the 7 std values
        self.std_layer = tf.keras.layers.Dense(7)

    def call(self, inputs, training=None):
        # 'inputs' is now a dictionary
        # Process each input with its specialized layer.
        gripper_encoding = self.gripper_encoder(inputs['gripper'])
        block_encoding = self.block_encoder(inputs['block'])
        goal_encoding = self.goal_encoder(inputs['goal'])

        # Combine the high-level features.
        combined_encoding = tf.concat(
            [gripper_encoding, block_encoding, goal_encoding], 
            axis=-1
        )

        # Pass combined features through shared layers.
        hidden = self.shared_hidden_1(combined_encoding)
        hidden = self.shared_hidden_2(hidden)

        # Calculate means and stds.
        means = self.mean_layer(hidden)
        stds = tf.nn.softplus(self.std_layer(hidden)) + 1e-6

        return means, stds