import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from collections import namedtuple
import os
import torch
#from stable_baselines3.common.env_checker import check_env
import pickle
from tqdm import tqdm

from custom_recommender.envs.custom_recommender_env import RecommenderEnv
from agent import Agent


if __name__ == "__main__":

    problems = pd.read_csv("/teamspace/studios/this_studio/recommender/custom_envs/data/problems.csv")
    solutions = pd.read_csv("/teamspace/studios/this_studio/recommender/custom_envs/data/submissions.csv")

    solutions = solutions.sort_values(by='id').reset_index(drop=True)


    # Set verdict_value to 0 where it doesn't equal 1
    solutions['verdict_value'] = solutions['verdict_value'].apply(lambda x: x if x == 1 else 0)


    # Load encoders
    with open('/teamspace/studios/this_studio/recommender/custom_envs/encoders/user_encoder.pkl', 'rb') as f:
        user_encoder = pickle.load(f)

    with open('/teamspace/studios/this_studio/recommender/custom_envs/encoders/problem_encoder.pkl', 'rb') as f:
        problem_encoder = pickle.load(f)

    solutions['user_id'] = user_encoder.fit_transform(solutions['user_id'])
    solutions['problem_id'] = problem_encoder.fit_transform(solutions['problem_id'])

    problems["problem_id"] = problem_encoder.fit_transform(problems['problem_id'])


    problems_dict = {}
    for index, row in problems.iterrows():
        problem_id = row['problem_id']
        difficulty = row['difficulty_score']
        tags = row['problem.tags'].split(',')
        problem_info = (difficulty, tags)
        problems_dict[problem_id] = problem_info

    users_dict = {}
    for index, row in solutions.iterrows():
        user_id = row['user_id']
        problem_id = row['problem_id']
        status = row['verdict_value']
        problem_status = (problem_id, status)
        if user_id in users_dict:
            users_dict[user_id].append(problem_status)
        else:
            users_dict[user_id] = [problem_status]

    problem_id_to_name = dict(zip(problems['problem_id'], problems['problem.name']))

    users_num = len(users_dict)
    problems_num = len(problems_dict)

    train_users_num = int(users_num * 0.8)
    train_users_dict = {k: users_dict.get(k) for k in list(users_dict.keys())[:train_users_num]}

    # Create the validation users dictionary
    validation_users_dict = {k: users_dict[k] for k in list(users_dict.keys())[train_users_num:]}

# Training loop
NUM_EPISODES = 700
MAX_STEPS = 300
ACTION_SIZE = len(problems_dict)
STATE_SIZE = 256

agent = Agent(input_size=STATE_SIZE, output_size=ACTION_SIZE)
env = RecommenderEnv(train_users_dict, problems_dict, 300, MAX_STEPS)
validation_env = RecommenderEnv(validation_users_dict, problems_dict, 300, MAX_STEPS)

training_rewards = []

# Log structure for training
training_log = []


for episode in tqdm(range(NUM_EPISODES), desc='Training', leave=False):
    state, _ = env.reset()
    total_reward = 0
    steps = 0

    for step in range(MAX_STEPS):
        action = agent.select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.replay_buffer.push(state, action, next_state, reward, done)
        agent.update_model()
        state = next_state
        total_reward += reward
        steps += 1

        if done or truncated:
            break

    training_rewards.append(total_reward)
    training_log.append((episode, steps, total_reward))
    agent.update_epsilon()


    if (episode + 1) % 50 == 0:
        print(f"Training - Episode: {episode}, Steps: {steps}, Total Reward: {total_reward}")

        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"model_episode_{episode + 1}.pt")
        torch.save(agent.policy_net.state_dict(), model_path)

# Save training log to a DataFrame
training_log_df = pd.DataFrame(training_log, columns=['Episode', 'Steps', 'Total Reward'])
training_log_df.to_csv("training_log.csv", index=False)

# Validation loop
def evaluate_agent(agent, env, num_episodes=200):
    agent.epsilon=0

    total_rewards = []
    validation_log = []

    for episode in tqdm(range(num_episodes), desc='Validation', leave=False):
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            steps += 1

            if done or truncated:
                break

        total_rewards.append(total_reward)
        validation_log.append((episode, steps, total_reward))
        if (episode + 1) % 50 == 0:
            print(f"Validation - Episode: {episode}, Steps: {steps}, Total Reward: {total_reward}")
    
    # Save validation log to a DataFrame
    validation_log_df = pd.DataFrame(validation_log, columns=['Episode', 'Steps', 'Total Reward'])
    validation_log_df.to_csv("validation_log.csv", index=False)
    
    return total_rewards

# Evaluate the agent
validation_rewards = evaluate_agent(agent, validation_env)


def hit_rate_at_k(recommended, actual, k):
    """
    Calculate Hit Rate at K (HR@K).
    
    :param recommended: List of lists containing the recommended items for each user.
    :param actual: List of lists containing the actual items for each user.
    :param k: Number of top recommendations to consider.
    :return: Hit Rate at K.
    """
    hits = 0
    num_users = len(actual)
    
    for i in range(num_users):
        correct_problems = [item[0] for item in actual[i] if item[1] == 1]
        hits += int(any(item in correct_problems for item in recommended[i][:k]))
    
    return hits / num_users


def mrr_at_k(recommended, actual, k):
    """
    Calculate Mean Reciprocal Rank at K (MRR@K).
    
    :param recommended: List of lists containing the recommended items for each user.
    :param actual: List of lists containing the actual items for each user, where each item is a tuple (problem_id, verdict).
    :param k: Number of top recommendations to consider.
    :return: Mean Reciprocal Rank at K.
    """
    mrr = 0
    num_users = len(actual)
    
    for i in range(num_users):
        correct_problems = [item[0] for item in actual[i] if item[1] == 1] 
        for rank, item in enumerate(recommended[i][:k], start=1):
            if item in correct_problems:
                mrr += 1 / rank
                break
    
    return mrr / num_users

def novelty_at_k(recommended, actual, k):
    """
    Calculate Novelty at K (Novelty@K).
    
    :param recommended: List of lists containing the recommended items for each user.
    :param actual: List of lists containing the actual items for each user, where each item is a tuple (problem_id, verdict).
    :param k: Number of top recommendations to consider.
    :return: Novelty at K.
    """
    novelty = 0
    num_users = len(actual)
    
    for i in range(num_users):
        # Collect problem ids that were answered correctly (verdict == 1)
        correct_problems = {item[0] for item in actual[i] if item[1] == 1}
        
        for item in recommended[i][:k]:
            # Check if the recommended item is not in the list of correctly solved problems
            if item not in correct_problems:
                novelty += 1
    
    return novelty / (num_users * k)



def evaluate_recommendations(agent, env, K=10, evaluation_proportion=0.3):
    all_recommended = []
    all_actual = []

    for user_id in tqdm(env.available_users, desc=f'Evaluating', leave=False):
        state, _ = env.reset(user_id=user_id, evaluation_mode=True)
        top_k_actions = agent.select_top_k_actions(state, K)
        all_recommended.append(top_k_actions)

        test_data_length = int(len(env.users_dict[user_id]) * evaluation_proportion)
        test_data = env.users_dict[user_id][-test_data_length:]
        test_problems = [x for x in test_data]
        all_actual.append(test_problems)

    hr_k = hit_rate_at_k(all_recommended, all_actual, K)
    mrr_k = mrr_at_k(all_recommended, all_actual, K)
    novelty_k = novelty_at_k(all_recommended, all_actual, K)

    return hr_k, mrr_k, novelty_k

# Usage:
hr_10, mrr_10, novelty_10 = evaluate_recommendations(agent, validation_env, 10)
print(f"HR@10: {hr_10}")
print(f"MRR@10: {mrr_10}")
print(f"Novelty@10: {novelty_10}")

hr_15, mrr_15, novelty_15 = evaluate_recommendations(agent, validation_env, 15)
print(f"HR@15: {hr_15}")
print(f"MRR@15: {mrr_15}")
print(f"Novelty@15: {novelty_15}")


#agent.visualize_training_results(training_rewards, validation_rewards, True)
