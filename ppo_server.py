from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys

app = Flask(__name__)

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
        return torch.softmax(self.fc3(x), dim=-1)  # Action probabilities


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
        self.policy = PolicyNetwork(state_dim=20, action_dim=8)
        self.value = ValueNetwork(state_dim=20)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=0.0003)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=0.001)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        # Load existing model if available
        self.load_model()
        
        print(f"✓ {team_name} agent ready", file=sys.stderr)
    
    def load_model(self):
        """Load weights from disk"""
        try:
            checkpoint = torch.load(self.model_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
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
    
    def select_action(self, state):
        """Choose action based on current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action_probs = self.policy(state_tensor)
        
        # Sample action from probability distribution
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample().item()
        
        return action
    
    def store_experience(self, state, action, reward, done):
        """Store experience for later training"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self):
        """Update neural network weights (PPO algorithm)"""
        print(f"UPDATE called for {self.team_name}", file=sys.stderr)
        print(f"Experiences in buffer: {len(self.states)}", file=sys.stderr)
        
        if len(self.states) == 0:
            print(f"⚠ No experiences to update for {self.team_name}", file=sys.stderr)
            # Save initial model anyway
            self.save_model()
            return
        
        print(f"Processing {len(self.states)} experiences")
        
        # Convert to tensors
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        rewards = torch.FloatTensor(self.rewards)
        
        # Calculate advantages
        with torch.no_grad():
            values = self.value(states).squeeze()
        advantages = rewards - values
        
        # PPO update: multiple epochs
        for epoch in range(10):
            # Policy update
            action_probs = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            
            policy_loss = -(log_probs * advantages).mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()
        
        # Value function update
        for epoch in range(10):
            values = self.value(states).squeeze()
            value_loss = ((values - rewards) ** 2).mean()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
        
        print(f"Policy loss: {policy_loss.item():.4f}")
        print(f"Value loss: {value_loss.item():.4f}")
        
        # Clear buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        
        # Auto-save after update
        self.save_model()
        print(f"✓ {self.team_name} model updated and saved")


# ========== INITIALIZE AGENTS (ONCE, AT STARTUP) ==========

print("=" * 80, file=sys.stderr)
print("PPO SERVER STARTING", file=sys.stderr)
print("=" * 80, file=sys.stderr)

team1_agent = PPOAgent(team_name="team1")
team2_agent = PPOAgent(team_name="team2")

# Store current state/action for each team (for reward association)
current_state = {}
current_action = {}

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
        "team2_experiences": len(team2_agent.states)
    })


@app.route('/select_action', methods=['POST'])
def select_action():
    """
    Select an action based on current game state
    
    Request body:
    {
        "team": "team1" or "team2",
        "state": {
            "player_health": 5000,
            "player_pips": 3,
            "opponent_health": 4500,
            "opponent_pips": 5,
            "round_number": 1,
            "cards_remaining": 50,
            "has_aura": false,
            "opponent_has_aura": false,
            "valid_actions": [1, 1, 1, 1, 1, 1, 1, 1]
        }
    }
    
    Response:
    {
        "action": 2
    }
    """
    try:
        data = request.json
        team = data.get('team', 'team2')
        state_data = data.get('state', {})
        
        # Select agent
        agent = team1_agent if team == "team1" else team2_agent
        
        # Extract and normalize state features
        state = [
            state_data.get('player_health', 12000) / 12000.0,      # Changed from 5000
            state_data.get('player_pips', 0) / 14.0,
            state_data.get('opponent_health', 12000) / 12000.0,    # Changed from 5000
            state_data.get('opponent_pips', 0) / 14.0,
            state_data.get('round_number', 1) / 50.0,              # Changed from 20 (with round limit at 50)
            state_data.get('cards_remaining', 50) / 100.0,
            float(state_data.get('has_aura', False)),
            float(state_data.get('opponent_has_aura', False)),
        ]
        
        # Pad to 20 features
        while len(state) < 20:
            state.append(0.0)
        
        # Store state for later reward association
        current_state[team] = state
        
        # Select action
        action = agent.select_action(state)
        
        # Store action for later reward association
        current_action[team] = action
        
        # Apply action mask (ensure valid action)
        valid_actions = state_data.get('valid_actions', [1] * 8)
        if len(valid_actions) > action and valid_actions[action] == 0:
            action = 0  # Default to pass if invalid
        
        return jsonify({"action": action})
    
    except Exception as e:
        print(f"Error in select_action: {e}", file=sys.stderr)
        return jsonify({"error": str(e), "action": 0}), 500


@app.route('/store_reward', methods=['POST'])
def store_reward():
    """
    Store reward for the last action taken
    
    Request body:
    {
        "team": "team1" or "team2",
        "reward": 8.5,
        "done": false
    }
    
    Response:
    {
        "status": "ok"
    }
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
            agent.store_experience(
                current_state[team],
                current_action[team],
                reward,
                done
            )
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
    
    Request body:
    {
        "team": "team1" or "team2"
    }
    
    Response:
    {
        "status": "updated"
    }
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
    
    Request body:
    {
        "team": "team1" or "team2",
        "path": "checkpoints/team1_match_10.pth" (optional)
    }
    
    Response:
    {
        "status": "saved"
    }
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
    print("Server starting on http://0.0.0.0:5000", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)