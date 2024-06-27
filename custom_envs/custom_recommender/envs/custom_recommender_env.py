from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
import gymnasium
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
from torch.nn import functional as F



# Load data
df = pd.read_csv("/teamspace/studios/this_studio/recommender/custom_envs/data/submissions.csv", usecols=['id', 'user_id', 'problem_id', 'problem.tags', 'verdict_value'])

# Order data ascendingly by id
df = df.sort_values(by='id').reset_index(drop=True)

# Set verdict_value to 0 where it doesn't equal 1
df['verdict_value'] = df['verdict_value'].apply(lambda x: x if x == 1 else 0)


# Load encoders
with open('/teamspace/studios/this_studio/recommender/custom_envs/encoders/user_encoder.pkl', 'rb') as f:
    user_encoder = pickle.load(f)

with open('/teamspace/studios/this_studio/recommender/custom_envs/encoders/problem_encoder.pkl', 'rb') as f:
    problem_encoder = pickle.load(f)

df['user_id'] = user_encoder.fit_transform(df['user_id'])
df['problem_id'] = problem_encoder.fit_transform(df['problem_id'])

# Load the GCN embeddings
gcn_embeddings = torch.load('/teamspace/studios/this_studio/recommender/custom_envs/graph embeddings/gcn_embeddings.pth')

class RecapModule(nn.Module):
    def __init__(self, hidden_dim):
        super(RecapModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_out):
        # Compute attention scores
        attention_scores = self.attention(lstm_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Compute weighted sum of historical exercises
        recap_out = torch.sum(attention_weights * lstm_out, dim=1)
        return recap_out

class KTModel(nn.Module):
    def __init__(self, num_problems, embed_dim, hidden_dim, output_dim, gcn_output_dim):
        super(KTModel, self).__init__()
        self.embedding = nn.Embedding(num_problems, embed_dim, padding_idx=0)
        self.linear = nn.Linear(gcn_output_dim + 1, embed_dim)  # Linear layer for concatenated graph-based embeddings and verdict value
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.recap_module = RecapModule(hidden_dim)
        self.output = nn.Linear(hidden_dim * 2, output_dim)
        #self.gcn = GCN(input_dim, hidden_dim, gcn_output_dim)

    def forward(self, problems, gcn_embeddings, verdicts):
        batch_size, seq_len = problems.size()
        problem_embeds = self.embedding(problems)
        gcn_embeddings_batch = gcn_embeddings[problems]

        verdicts = verdicts.unsqueeze(-1).float()  # Convert verdicts to float and add a dimension
        concat_features = torch.cat((gcn_embeddings_batch, verdicts), dim=-1)
        transformed_features = F.relu(self.linear(concat_features))

        lstm_out, _ = self.lstm(transformed_features)
        recap_out = self.recap_module(lstm_out)
        combined_out = torch.cat((lstm_out, recap_out.unsqueeze(1).repeat(1, seq_len, 1)), dim=-1)
        output = self.output(combined_out).squeeze(-1)
        return torch.sigmoid(output), combined_out
    
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X, adj):
        X = torch.mm(adj, X)
        X = self.linear(X)
        return F.relu(X)

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, output_dim)

    def forward(self, X, adj):
        X = self.layer1(X, adj)
        X = self.layer2(X, adj)
        return X


class RecommenderEnv(gymnasium.Env):
    def __init__(self, users_dict, problems_dict, min_trajectory=200, max_steps=20, reward_params=None):

        self.evaluation_mode = False  # Flag to indicate if the environment is in evaluation mode

        # user training data dictionary containing
        # key: user id
        # value: a list of (assignment id, status) tuples
        # the status is the feedback recieved on the user's solution to the problem
        self.users_dict = users_dict

        # problems dictionary containing all the problems in the dataset
        # key: problem id
        # value: a (problem representation, difiiculty score, tags) tuple
        # the problem representation is a vector representation for the important problem attributes (problem statement, difficulty, tags)
        self.problems_dict = problems_dict

        # the minimum length of user trajectory for the data that will be used for training
        self.min_trajectory = min_trajectory

        self.max_steps = max_steps

        # users with the required minimum trajectory length
        self.available_users = self._generate_available_users()
        


        # Default reward parameters, can be overridden by reward_params argument
        if reward_params is None:
            reward_params = {
                'tau1': 0.7,
                'tau2': 0.3,
                'tau3': -0.5,
                'eta1': 0.6,
                'eta2': -0.6,
                'beta1': 1.0,
                'beta2': 1.0,
                'beta3': 1.0
            }
        self.reward_params = reward_params
        
        # an action is to recommend a problem so the action space would be all the problems
        self.action_space = Discrete(len(problems_dict)) 

        # the dkt model hidden state dimension
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(256,), dtype=np.float32)  


        self.current_user_id = None
        self.current_user_history = None
        self.current_step = 0
        self.min_start = 50
        self.recommended_problems = set()
        self.done = False

        # dkt 
        num_problems = len(problem_encoder.classes_)
        embed_dim = 32
        hidden_dim = 128
        output_dim = 1
        gcn_output_dim = 32

        self.dkt_model = KTModel(num_problems, embed_dim, hidden_dim, output_dim, gcn_output_dim)
        self.dkt_model.load_state_dict(torch.load('/teamspace/studios/this_studio/recommender/custom_envs/model weights/kt_model_weights.pth'))
        self.dkt_model.eval()



    # filters out users based on exercise history length
    def _generate_available_users(self):
        available_users = []
        for user_id, history in self.users_dict.items():
            if history is not None and len(history) >= self.min_trajectory:
                available_users.append(user_id)
        print(f"Total available users: {len(available_users)}")
        return available_users


    def reset(self, seed=None, user_id=None, evaluation_mode=False):

        self.evaluation_mode = evaluation_mode  # Set the evaluation mode
        self.min_start = len(self.current_user_history) if self.evaluation_mode else 50


        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if user_id is None:
            self.current_user_id = random.choice(self.available_users)
        else:
            self.current_user_id = user_id

        self.current_user_history = self.users_dict[self.current_user_id]
        self.current_step = 0
        self.recommended_problems = set()
        self.done = False
                
        state = self.get_state()
        return state.detach().numpy(), {}  # Return a tuple with default values for reward, done, and info

    def step(self, action):
        recommended_problem_id = list(self.problems_dict.keys())[action]

        self.recommended_problems.add(recommended_problem_id)
        reward = self.calculate_reward(recommended_problem_id)

        self.current_step += 1
        self.done = self.current_step >= len(self.current_user_history)
        
        state = self.get_state()
        truncated = self.current_step >= self.max_steps
        terminated = self.done and not truncated
        return state.detach().numpy(), reward, terminated, truncated, {}
    
    def get_state(self):
        user_id = self.current_user_id

        # Retrieve the group of data for the current user
        group = df.groupby('user_id').get_group(user_id)

        # Sample problems and verdicts up to the given index
        history_length = len(group) if self.evaluation_mode else self.min_start
        sampled_problems = group['problem_id'][:history_length].values
        sampled_verdicts = group['verdict_value'][:history_length].values

        # Convert problem IDs and verdicts to tensors
        problem_ids = torch.tensor(sampled_problems, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        verdicts = torch.tensor(sampled_verdicts, dtype=torch.float).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs, hidden = self.dkt_model(problem_ids, gcn_embeddings, verdicts)

        # Extract the last time step from the hidden state
        state = hidden[:, -1, :].squeeze(0)  # Shape will be [256,]

        if not self.evaluation_mode:
            self.min_start += 1

        return state

    def calculate_reward(self, recommended_problem_id):
        current_problem_id, current_status = self.current_user_history[self.current_step]
        current_problem_info = self.problems_dict[current_problem_id]
        recommended_problem_info = self.problems_dict[recommended_problem_id]
        
        r1 = self.review_and_discover(current_status, current_problem_info, recommended_problem_info)
        r2 = self.knowledge_point_smoothness(current_problem_info, recommended_problem_info)
        r3 = self.difficulty_smoothness(current_problem_info, recommended_problem_info)
        
        beta1 = self.reward_params['beta1']
        beta2 = self.reward_params['beta2']
        beta3 = self.reward_params['beta3']
        reward = beta1 * r1 + beta2 * r2 + beta3 * r3
        return reward

    def review_and_discover(self, current_status, current_problem_info, recommended_problem_info):
        tau1 = self.reward_params['tau1']
        tau2 = self.reward_params['tau2']
        tau3 = self.reward_params['tau3']
        kt = current_problem_info[1]
        kt1 = recommended_problem_info[1]
        
        if current_status == 1:
            if not set(kt).intersection(set(kt1)):
                return tau1
            else:
                return tau2
        else:
            if not set(kt).intersection(set(kt1)):
                return tau3
            else:
                return 0

    def knowledge_point_smoothness(self, current_problem_info, recommended_problem_info):
        eta1 = self.reward_params['eta1']
        eta2 = self.reward_params['eta2']
        kt = current_problem_info[1]
        kt1 = recommended_problem_info[1]
        nkt1_kt = abs(len(kt1) - len(kt))
        
        if nkt1_kt == 1 and set(kt).intersection(set(kt1)):
            return eta1
        elif nkt1_kt > 1:
            return eta2
        else:
            return 0

    def difficulty_smoothness(self, current_problem_info, recommended_problem_info):
        dt = float(current_problem_info[0])  
        dt1 = float(recommended_problem_info[0]) 

        difficulty_diff = dt1 - dt

        reward = (4 / np.pi) * (np.arctan(-(difficulty_diff**2)) + (np.pi / 4))
        return reward
