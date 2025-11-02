import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import pickle
import os

class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        shared_out = self.shared(state)
        action_probs = self.actor(shared_out)
        state_value = self.critic(shared_out)
        return action_probs, state_value


class PPOAgent:
    """PPO Agent for Wizard101-style card game"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, 
                 c1=0.5, c2=0.01, update_epochs=10, model_path='ppo_model.pth'):
        self.gamma = gamma
        self.epsilon = epsilon  # PPO clipping parameter
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        self.update_epochs = update_epochs
        self.model_path = model_path
        
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Memory for storing trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Load model if exists
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def select_action(self, state, valid_actions_mask=None, training=True):
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs, state_value = self.network(state_tensor)
        
        # Mask invalid actions
        if valid_actions_mask is not None:
            mask = torch.FloatTensor(valid_actions_mask)
            action_probs = action_probs * mask
            action_probs = action_probs / action_probs.sum()  # Renormalize
        
        if training:
            # Sample from distribution during training
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store for learning
            self.states.append(state)
            self.actions.append(action.item())
            self.log_probs.append(log_prob.item())
            self.values.append(state_value.item())
            
            return action.item()
        else:
            # Take best action during evaluation
            return torch.argmax(action_probs).item()
    
    def store_reward(self, reward, done):
        """Store reward and done flag"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self):
        """Compute returns and advantages using GAE"""
        returns = []
        advantages = []
        gae = 0
        
        # Add dummy value for terminal state
        values = self.values + [0]
        
        # Compute advantages using GAE (lambda=0.95)
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * values[i + 1] * (1 - self.dones[i]) - values[i]
            gae = delta + self.gamma * 0.95 * (1 - self.dones[i]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return returns, advantages
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return 0.0
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0.0
        
        # PPO update for multiple epochs
        for _ in range(self.update_epochs):
            # Get current policy outputs
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Compute ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Compute clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss = nn.MSELoss()(values.squeeze(), returns)
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        return total_loss / self.update_epochs
    
    def save_model(self, path=None):
        """Save model to disk"""
        if path is None:
            path = self.model_path
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path=None):
        """Load model from disk"""
        if path is None:
            path = self.model_path
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.network.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def parse_game_state(state_json):
    """Parse game state from JSON and convert to feature vector"""
    state = json.loads(state_json)
    
    # Extract features from game state
    features = []
    
    # Player features
    features.append(state.get('player_health', 0) / 10000.0)  # Normalize
    features.append(state.get('player_pips', 0) / 14.0)
    features.append(state.get('player_shadow_gauge', 0) / 200.0)
    
    # Opponent features
    features.append(state.get('opponent_health', 0) / 10000.0)
    features.append(state.get('opponent_pips', 0) / 14.0)
    features.append(state.get('opponent_shadow_gauge', 0) / 200.0)
    
    # Hand information (7 cards max)
    hand = state.get('hand', [])
    for i in range(7):
        if i < len(hand):
            card = hand[i]
            features.append(1.0 if card.get('is_valid', False) else 0.0)
            features.append(card.get('pips', 0) / 14.0)
            features.append(card.get('damage_potential', 0) / 5000.0)
        else:
            features.extend([0.0, 0.0, 0.0])
    
    # Active effects/auras
    features.append(1.0 if state.get('has_aura', False) else 0.0)
    features.append(1.0 if state.get('opponent_has_aura', False) else 0.0)
    
    # Combat statistics
    features.append(state.get('round_number', 0) / 50.0)  # Normalize by max expected rounds
    features.append(state.get('cards_remaining', 0) / 64.0)  # Typical deck size
    
    return np.array(features, dtype=np.float32)


def main():
    """Main entry point for PPO agent"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No command provided"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Initialize agent (state_dim based on feature vector, action_dim = 8: 0=pass, 1-7=cast card)
    state_dim = 31  # Matches feature vector size
    action_dim = 8  # 0 = pass, 1-7 = cast card from hand
    agent = PPOAgent(state_dim, action_dim)
    
    if command == "select_action":
        # Read game state from stdin or argument
        if len(sys.argv) > 2:
            state_json = sys.argv[2]
        else:
            state_json = sys.stdin.read()
        
        # Parse state
        state_features = parse_game_state(state_json)
        
        # Parse valid actions mask if provided
        state = json.loads(state_json)
        valid_actions = state.get('valid_actions', [1] * action_dim)
        
        # Select action
        training_mode = state.get('training', True)
        action = agent.select_action(state_features, valid_actions, training=training_mode)
        
        # Return action
        result = {"action": action}
        print(json.dumps(result))
    
    elif command == "store_reward":
        # Store reward for last action
        if len(sys.argv) > 3:
            reward = float(sys.argv[2])
            done = sys.argv[3].lower() == 'true'
        else:
            data = json.loads(sys.stdin.read())
            reward = data.get('reward', 0.0)
            done = data.get('done', False)
        
        agent.store_reward(reward, done)
        print(json.dumps({"status": "reward_stored"}))
    
    elif command == "update":
        # Perform PPO update
        loss = agent.update()
        print(json.dumps({"status": "updated", "loss": loss}))
    
    elif command == "save_model":
        # Save model
        path = sys.argv[2] if len(sys.argv) > 2 else None
        agent.save_model(path)
        print(json.dumps({"status": "model_saved"}))
    
    elif command == "load_model":
        # Load model
        path = sys.argv[2] if len(sys.argv) > 2 else None
        agent.load_model(path)
        print(json.dumps({"status": "model_loaded"}))
    
    else:
        print(json.dumps({"error": f"Unknown command: {command}"}))
        sys.exit(1)


if __name__ == "__main__":
    main()