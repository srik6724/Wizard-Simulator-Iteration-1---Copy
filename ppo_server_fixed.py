
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
import math
import random

app = Flask(__name__)

# ========== UTILS ==========

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== NEURAL NETWORK CLASSES ==========

class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities
    """
    def __init__(self, state_dim=20, action_dim=8):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return torch.softmax(logits, dim=-1)  # Action probabilities


class ValueNetwork(nn.Module):
    """
    Value network that estimates state values
    """
    def __init__(self, state_dim=20):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)  # Single value output
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # State value estimate


# ========== PPO AGENT CLASS ==========

class PPOAgent:
    """
    PPO agent that stays in memory (no reloading every call)
    """
    def __init__(self, team_name="team1"):
        print(f"Initializing {team_name} agent...", file=sys.stderr)
        
        self.team_name = team_name
        self.model_path = f'ppo_model_{team_name}.pth'
        
        # Create networks
        self.policy = PolicyNetwork(state_dim=20, action_dim=8).to(DEVICE)
        self.value = ValueNetwork(state_dim=20).to(DEVICE)
        self.policy.train(); self.value.train()
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.vals = []
        
        # Load existing model if available
        self.load_model()
        
        print(f"✓ {team_name} agent ready", file=sys.stderr)
    
    def load_model(self):
        """Load weights from disk"""
        try:
            checkpoint = torch.load(self.model_path, map_location='cpu')
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            if 'policy_optimizer' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            if 'value_optimizer' in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            print(f"✓ Loaded model from {self.model_path}", file=sys.stderr)
        except FileNotFoundError:
            print(f"⚠ No existing model. Starting fresh for {self.team_name}", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Error loading model: {e}", file=sys.stderr)
    
    def save_model(self, path=None):
        """Save weights to disk"""
        if path is None:
            path = self.model_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
            }, path)
            print(f"✓ Saved model to {path}", file=sys.stderr)
        except Exception as e:
            print(f"✗ Error saving model: {e}", file=sys.stderr)
    
    @torch.no_grad()
    def select_action(self, state, valid_actions=None):
        """Choose action based on current policy with action masking"""
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        probs = self.policy(state_tensor).squeeze(0)  # [A]
        
        # Apply mask if provided
        if valid_actions is not None and len(valid_actions) == probs.shape[-1]:
            mask = torch.tensor(valid_actions, dtype=torch.float32, device=DEVICE)
            # ensure no all-zero mask
            if mask.sum().item() == 0:
                mask = torch.ones_like(mask)
            eps = 1e-8
            probs = probs * mask + eps  # zero out invalid actions
            probs = probs / probs.sum()  # renormalize
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value(state_tensor).squeeze(0).squeeze(-1)
        
        # Return Python scalars
        return int(action.item()), float(logprob.item()), float(value.item())
    
    def store_experience(self, state, action, reward, done, logprob, value):
        """Store experience for later training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.logprobs.append(logprob)
        self.vals.append(value)
    
    def _compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        T = len(rewards)
        adv = torch.zeros(T, dtype=torch.float32, device=DEVICE)
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + gamma * next_value * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + values
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns
    
    def update(self,
               gamma=0.99,
               lam=0.95,
               clip_eps=0.2,
               policy_epochs=10,
               value_epochs=10,
               entropy_coef=0.01):
        """Update neural network weights using PPO-Clip"""
        print(f"UPDATE called for {self.team_name}", file=sys.stderr)
        print(f"Experiences in buffer: {len(self.states)}", file=sys.stderr)
        
        if len(self.states) == 0:
            print(f"⚠ No experiences to update for {self.team_name}", file=sys.stderr)
            # Save initial model anyway
            self.save_model()
            return
        
        print(f"Processing {len(self.states)} experiences", file=sys.stderr)
        
        # Convert to tensors
        states = torch.tensor(self.states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)
        dones   = torch.tensor(self.dones, dtype=torch.float32, device=DEVICE)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=DEVICE)
        values = torch.tensor(self.vals, dtype=torch.float32, device=DEVICE)
        
        # Compute GAE and returns
        advantages, returns = self._compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        
        # Policy update
        for epoch in range(policy_epochs):
            probs = self.policy(states)               # [T, A]
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(actions)     # [T]
            entropy = dist.entropy().mean()
            
            ratio = (new_logprobs - old_logprobs).exp()
            unclipped = ratio * advantages
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            policy_loss = -(torch.min(unclipped, clipped).mean() + entropy_coef * entropy)
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # Value function update
        for epoch in range(value_epochs):
            value_pred = self.value(states).squeeze(-1)  # [T]
            value_loss = ((value_pred - returns) ** 2).mean()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        print(f"Policy loss: {policy_loss.item():.4f}", file=sys.stderr)
        print(f"Value loss: {value_loss.item():.4f}", file=sys.stderr)
        
        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.vals.clear()
        
        # Auto-save after update
        self.save_model()
        print(f"✓ {self.team_name} model updated and saved", file=sys.stderr)


# ========== INITIALIZE AGENTS (ONCE, AT STARTUP) ==========

print("=" * 80, file=sys.stderr)
print("PPO SERVER STARTING", file=sys.stderr)
print("=" * 80, file=sys.stderr)

team1_agent = PPOAgent(team_name="team1")
team2_agent = PPOAgent(team_name="team2")

# Store current state/action for each team (for reward association)
current_state = {}
current_action = {}  # will hold tuple (action, logprob, value)

print("=" * 80, file=sys.stderr)
print("✓ ALL AGENTS LOADED AND READY", file=sys.stderr)
print("=" * 80, file=sys.stderr)


# ========== HTTP ENDPOINTS ==========

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "team1_experiences": len(team1_agent.states),
        "team2_experiences": len(team2_agent.states),
        "device": str(DEVICE)
    })


@app.route('/select_action', methods=['POST'])
def select_action():
    """
    Select an action based on current game state
    """
    try:
        data = request.json
        team = data.get('team', 'team2')
        state_data = data.get('state', {})
        
        # Select agent
        agent = team1_agent if team == "team1" else team2_agent
        
        # Extract and normalize state features
        state = [
            state_data.get('player_health', 12000) / 12000.0,      # normalized to [0,1]
            state_data.get('player_pips', 0) / 14.0,
            state_data.get('opponent_health', 12000) / 12000.0,
            state_data.get('opponent_pips', 0) / 14.0,
            state_data.get('round_number', 1) / 50.0,
            state_data.get('cards_remaining', 50) / 100.0,
            float(state_data.get('has_aura', False)),
            float(state_data.get('opponent_has_aura', False)),
        ]
        
        # Pad to 20 features
        while len(state) < 20:
            state.append(0.0)
        
        # Mask (ensure valid action)
        valid_actions = state_data.get('valid_actions', [1] * 8)
        # Select action with masking applied *inside* the agent
        action, logprob, value = agent.select_action(state, valid_actions=valid_actions)
        
        # Store state and action tuple for later reward association
        current_state[team] = state
        current_action[team] = (action, logprob, value)
        
        return jsonify({"action": int(action)})
    
    except Exception as e:
        print(f"Error in select_action: {e}", file=sys.stderr)
        return jsonify({"error": str(e), "action": 0}), 500


@app.route('/store_reward', methods=['POST'])
def store_reward():
    """
    Store reward for the last action taken
    """
    try:
        data = request.json
        team = data.get('team', 'team2')
        reward = data.get('reward', 0.0)
        done = data.get('done', False)
        
        # Select agent
        agent = team1_agent if team == "team1" else team2_agent
        
        # Store experience if we have state and action
        if team in current_state and team in current_action:
            action, logprob, value = current_action[team]
            agent.store_experience(
                current_state[team],
                action,
                reward,
                done,
                logprob,
                value
            )
            # If done, clear the held state/action for that team
            if done:
                current_state.pop(team, None)
                current_action.pop(team, None)
            return jsonify({"status": "ok"})
        else:
            print(f"⚠ WARNING: No state/action stored for {team}", file=sys.stderr)
            return jsonify({"status": "warning", "message": "No state/action found"}), 400
    
    except Exception as e:
        print(f"Error in store_reward: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route('/update', methods=['POST'])
def update():
    """
    Update the agent (train on stored experiences)
    """
    try:
        data = request.json
        team = data.get('team', 'team2')
        
        # Select agent
        agent = team1_agent if team == "team1" else team2_agent
        
        # Perform update
        agent.update()
        
        return jsonify({"status": "updated"})
    
    except Exception as e:
        print(f"Error in update: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route('/save_model', methods=['POST'])
def save_model():
    """
    Save the model to a specific path
    """
    try:
        data = request.json
        team = data.get('team', 'team2')
        path = data.get('path', None)
        
        # Select agent
        agent = team1_agent if team == "team1" else team2_agent
        
        # Save model
        agent.save_model(path)
        
        return jsonify({"status": "saved"})
    
    except Exception as e:
        print(f"Error in save_model: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


# ========== START SERVER ==========

if __name__ == '__main__':
    print("\n" + "=" * 80, file=sys.stderr)
    print("🚀 PPO SERVER READY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("Endpoints:", file=sys.stderr)
    print("  POST /select_action - Get action from agent", file=sys.stderr)
    print("  POST /store_reward  - Store experience", file=sys.stderr)
    print("  POST /update        - Train agent", file=sys.stderr)
    print("  POST /save_model    - Save model checkpoint", file=sys.stderr)
    print("  GET  /health        - Health check", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Device: {DEVICE}", file=sys.stderr)
    print("Server starting on http://0.0.0.0:5000", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
