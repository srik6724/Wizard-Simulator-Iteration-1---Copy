from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
import sys
import math
import random
import numpy as np
import gc  # For garbage collection

app = Flask(__name__)

# ========== UTILS ==========

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== IMPROVED NEURAL NETWORK CLASSES ==========

class PolicyNetwork(nn.Module):
    """
    Enhanced policy network with larger capacity for complex game states
    """
    def __init__(self, state_dim=50, action_dim=8, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, action_dim)
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        logits = self.fc4(x)
        return torch.softmax(logits, dim=-1)


class ValueNetwork(nn.Module):
    """
    Enhanced value network with larger capacity
    """
    def __init__(self, state_dim=50, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, 1)
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = torch.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


# ========== ENHANCED PPO AGENT CLASS WITH MEMORY FIX ==========

class PPOAgent:
    """
    PPO agent with improved training dynamics and MEMORY LEAK FIX
    """
    def __init__(self, team_name="team1", state_dim=50):
        print(f"Initializing {team_name} agent...", file=sys.stderr)
        
        self.team_name = team_name
        self.model_path = f'ppo_model_{team_name}.pth'
        self.state_dim = state_dim
        
        # Create networks with larger capacity
        self.policy = PolicyNetwork(state_dim=state_dim, action_dim=8, hidden_dim=256).to(DEVICE)
        self.value = ValueNetwork(state_dim=state_dim, hidden_dim=256).to(DEVICE)
        self.policy.train(); self.value.train()
        
        # Optimizers with learning rate scheduling
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)
        
        # Learning rate schedulers
        self.policy_scheduler = optim.lr_scheduler.StepLR(
            self.policy_optimizer, step_size=50, gamma=0.95
        )
        self.value_scheduler = optim.lr_scheduler.StepLR(
            self.value_optimizer, step_size=50, gamma=0.95
        )
        
        # Experience buffer with MAX SIZE to prevent memory leak
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logprobs = []
        self.vals = []
        self.max_buffer_size = 1000  # MEMORY FIX: Limit buffer size
        
        # Metrics tracking
        self.episode_rewards = []
        self.total_updates = 0
        self.wins = 0
        self.losses = 0
        
        # Load existing model if available
        self.load_model()
        
        print(f"✓ {team_name} agent ready (state_dim={state_dim})", file=sys.stderr)
    
    def load_model(self):
        """Load weights from disk"""
        try:
            checkpoint = torch.load(self.model_path, map_location=DEVICE)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.value.load_state_dict(checkpoint['value_state_dict'])
            if 'policy_optimizer' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            if 'value_optimizer' in checkpoint:
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
            if 'total_updates' in checkpoint:
                self.total_updates = checkpoint['total_updates']
            if 'wins' in checkpoint:
                self.wins = checkpoint['wins']
            if 'losses' in checkpoint:
                self.losses = checkpoint['losses']
            print(f"✓ Loaded model from {self.model_path}", file=sys.stderr)
            print(f"  Updates: {self.total_updates}, W/L: {self.wins}/{self.losses}", file=sys.stderr)
        except FileNotFoundError:
            print(f"⚠ No existing model. Starting fresh for {self.team_name}", file=sys.stderr)
        except Exception as e:
            print(f"⚠ Error loading model: {e}", file=sys.stderr)
    
    def save_model(self, path=None):
        """Save weights to disk"""
        if path is None:
            path = self.model_path
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        try:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'value_state_dict': self.value.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
                'total_updates': self.total_updates,
                'wins': self.wins,
                'losses': self.losses,
            }, path)
            print(f"✓ Saved model to {path}", file=sys.stderr)
        except Exception as e:
            print(f"✗ Error saving model: {e}", file=sys.stderr)
    
    @torch.no_grad()
    def select_action(self, state, valid_actions=None):
        """Choose action based on current policy with IMPROVED action masking"""
        # Ensure state has correct dimension
        if len(state) < self.state_dim:
            state = state + [0.0] * (self.state_dim - len(state))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
    
        state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        probs = self.policy(state_tensor).squeeze(0)
    
        # IMPROVED ACTION MASKING
        if valid_actions is not None and len(valid_actions) == probs.shape[-1]:
            mask = torch.tensor(valid_actions, dtype=torch.float32, device=DEVICE)
        
            # Check if mask is all zeros (no valid actions)
            if mask.sum().item() == 0:
                print("⚠ WARNING: All actions masked! Defaulting to action 0 (pass)", file=sys.stderr)
                return 0, 0.0, 0.0
        
            # CRITICAL FIX: Zero out invalid actions BEFORE applying
            masked_probs = probs * mask
        
            # Check if masking resulted in all zeros (numerical issue)
            if masked_probs.sum().item() < 1e-9:
                print("⚠ WARNING: Masked probs sum to zero! Valid actions:", 
                [i for i, v in enumerate(valid_actions) if v == 1], file=sys.stderr)
                # Force uniform distribution over valid actions
                masked_probs = mask / mask.sum()
            else:
                # Renormalize
                masked_probs = masked_probs / masked_probs.sum()
        
            probs = masked_probs
    
        # Sample from masked distribution
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        value = self.value(state_tensor).squeeze(0).squeeze(-1)
    
        # DEBUG: Print selected action and its probability
        action_int = int(action.item())
        if valid_actions is not None and action_int < len(valid_actions):
            if valid_actions[action_int] == 0:
                print(f"🔴 ERROR: Selected invalid action {action_int}! Mask: {valid_actions}", file=sys.stderr)
                print(f"   Probs before mask: {probs.tolist()}", file=sys.stderr)
    
        return action_int, float(logprob.item()), float(value.item())
    
    def store_experience(self, state, action, reward, done, logprob, value):
        """Store experience for later training with buffer size limit"""
        # Ensure state has correct dimension
        if len(state) < self.state_dim:
            state = state + [0.0] * (self.state_dim - len(state))
        elif len(state) > self.state_dim:
            state = state[:self.state_dim]
        
        # MEMORY FIX: Check buffer size before adding
        if len(self.states) >= self.max_buffer_size:
            print(f"⚠ Buffer full ({self.max_buffer_size}), forcing update...", file=sys.stderr)
            self.update()  # Force update to clear buffer
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.logprobs.append(logprob)
        self.vals.append(value)
        
        # Track episode rewards
        if done:
            episode_reward = sum(self.rewards)
            self.episode_rewards.append(episode_reward)
            
            # Keep only last 100 episode rewards to save memory
            if len(self.episode_rewards) > 100:
                self.episode_rewards = self.episode_rewards[-100:]
            
            # Track wins/losses (assume positive final reward = win)
            if reward > 5.0:  # Win bonus is 10.0, so final reward should be > 5
                self.wins += 1
            elif reward < -5.0:  # Loss penalty is -10.0
                self.losses += 1
    
    def _compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
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
        
        # Normalize advantages
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
            self.save_model()
            return
        
        # Convert to tensors
        states = torch.tensor(self.states, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(self.actions, dtype=torch.long, device=DEVICE)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=DEVICE)
        dones   = torch.tensor(self.dones, dtype=torch.float32, device=DEVICE)
        old_logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=DEVICE)
        values = torch.tensor(self.vals, dtype=torch.float32, device=DEVICE)
        
        # Compute GAE and returns
        advantages, returns = self._compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        
        # Track metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # Policy update with mini-batch training
        batch_size = min(64, len(states))
        indices = list(range(len(states)))
        
        for epoch in range(policy_epochs):
            random.shuffle(indices)
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                
                # Forward pass
                probs = self.policy(batch_states)
                dist = torch.distributions.Categorical(probs)
                new_logprobs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # PPO loss
                ratio = (new_logprobs - batch_old_logprobs).exp()
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
                policy_loss = -(torch.min(unclipped, clipped).mean() + entropy_coef * entropy)
                
                # Update
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_entropy += entropy.item()
        
        # Value function update with mini-batch training
        for epoch in range(value_epochs):
            random.shuffle(indices)
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                value_pred = self.value(batch_states).squeeze(-1)
                value_loss = ((value_pred - batch_returns) ** 2).mean()
                
                # Update
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value.parameters(), 0.5)
                self.value_optimizer.step()
                
                total_value_loss += value_loss.item()
        
        # Update learning rates
        self.policy_scheduler.step()
        self.value_scheduler.step()
        
        # Track metrics
        self.total_updates += 1
        avg_policy_loss = total_policy_loss / (policy_epochs * max(1, len(states) // batch_size))
        avg_value_loss = total_value_loss / (value_epochs * max(1, len(states) // batch_size))
        avg_entropy = total_entropy / (policy_epochs * max(1, len(states) // batch_size))
        
        print(f"Policy loss: {avg_policy_loss:.4f}", file=sys.stderr)
        print(f"Value loss: {avg_value_loss:.4f}", file=sys.stderr)
        print(f"Entropy: {avg_entropy:.4f}", file=sys.stderr)
        print(f"Total updates: {self.total_updates}", file=sys.stderr)
        print(f"W/L: {self.wins}/{self.losses} ({self.wins/(self.wins+self.losses+1e-8)*100:.1f}%)", file=sys.stderr)
        
        if len(self.episode_rewards) > 0:
            print(f"Avg episode reward (last 10): {np.mean(self.episode_rewards[-10:]):.2f}", file=sys.stderr)
        
        # MEMORY FIX: Clear buffer explicitly
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.logprobs.clear()
        self.vals.clear()
        
        # MEMORY FIX: Explicitly delete tensors and run garbage collection
        del states, actions, rewards, dones, old_logprobs, values
        del advantages, returns
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        # Auto-save after update
        self.save_model()
        print(f"✓ {self.team_name} model updated and saved", file=sys.stderr)


# ========== INITIALIZE AGENTS (ONCE, AT STARTUP) ==========

print("=" * 80, file=sys.stderr)
print("MEMORY-OPTIMIZED PPO SERVER STARTING", file=sys.stderr)
print("=" * 80, file=sys.stderr)

# Use larger state dimension to accommodate all features
STATE_DIM = 50

team1_agent = PPOAgent(team_name="team1", state_dim=STATE_DIM)
team2_agent = PPOAgent(team_name="team2", state_dim=STATE_DIM)

current_state = {}
current_action = {}

print("=" * 80, file=sys.stderr)
print("✓ ALL AGENTS LOADED AND READY", file=sys.stderr)
print("=" * 80, file=sys.stderr)


# ========== HTTP ENDPOINTS ==========

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # MEMORY FIX: Include buffer sizes in health check
    return jsonify({
        "status": "healthy",
        "team1_experiences": len(team1_agent.states),
        "team2_experiences": len(team2_agent.states),
        "team1_buffer_pct": f"{len(team1_agent.states)/team1_agent.max_buffer_size*100:.1f}%",
        "team2_buffer_pct": f"{len(team2_agent.states)/team2_agent.max_buffer_size*100:.1f}%",
        "team1_updates": team1_agent.total_updates,
        "team2_updates": team2_agent.total_updates,
        "team1_wl": f"{team1_agent.wins}/{team1_agent.losses}",
        "team2_wl": f"{team2_agent.wins}/{team2_agent.losses}",
        "device": str(DEVICE),
        "state_dim": STATE_DIM
    })


@app.route('/select_action', methods=['POST'])
def select_action():
    """Select an action based on current game state"""
    try:
        data = request.json
        team = data.get('team', 'team2')
        state_data = data.get('state', {})
        
        agent = team1_agent if team == "team1" else team2_agent
        
        # Extract and normalize ALL features
        state = [
            # Core stats (normalized)
            state_data.get('player_health', 11443) / 11443.0,
            state_data.get('player_health_pct', 1.0),
            state_data.get('player_pips', 0) / 14.0,
            state_data.get('player_shadow_gauge', 0) / 200.0,
            state_data.get('opponent_health', 11443) / 11443.0,
            state_data.get('opponent_health_pct', 1.0),
            state_data.get('opponent_pips', 0) / 14.0,
            state_data.get('opponent_shadow_gauge', 0) / 200.0,
            
            # Tactical state (normalized)
            state_data.get('player_blade_count', 0) / 5.0,
            state_data.get('player_blade_strength', 0) / 500.0,
            state_data.get('player_weakness_count', 0) / 5.0,
            state_data.get('player_weakness_strength', 0) / 300.0,
            state_data.get('player_shield_count', 0) / 5.0,
            state_data.get('player_shield_strength', 0) / 500.0,
            state_data.get('player_trap_count', 0) / 5.0,
            
            state_data.get('opponent_blade_count', 0) / 5.0,
            state_data.get('opponent_blade_strength', 0) / 500.0,
            state_data.get('opponent_weakness_count', 0) / 5.0,
            state_data.get('opponent_shield_count', 0) / 5.0,
            state_data.get('opponent_shield_strength', 0) / 500.0,
            state_data.get('opponent_trap_count', 0) / 5.0,
            state_data.get('opponent_trap_strength', 0) / 500.0,
            
            # Auras and effects
            float(state_data.get('player_has_aura', 0)),
            state_data.get('player_aura_strength', 0) / 50.0,
            state_data.get('player_aura_rounds_left', 0) / 4.0,
            float(state_data.get('opponent_has_aura', 0)),
            state_data.get('opponent_aura_rounds_left', 0) / 4.0,
            float(state_data.get('player_has_infallible', 0)),
            
            # Hand information
            state_data.get('castable_card_count', 0) / 7.0,
            state_data.get('total_damage_potential', 0) / 5000.0,
            
            # Resources
            state_data.get('cards_remaining', 50) / 100.0,
            state_data.get('opponent_cards_remaining', 50) / 100.0,
            
            # Game context
            state_data.get('round_number', 1) / 50.0,
            state_data.get('health_advantage', 0.0),
            state_data.get('pip_advantage', 0) / 14.0,
            state_data.get('tactical_advantage', 0) / 500.0,
        ]
        
        # Pad to STATE_DIM
        while len(state) < STATE_DIM:
            state.append(0.0)
        
        valid_actions = state_data.get('valid_actions', [1] * 8)
        action, logprob, value = agent.select_action(state, valid_actions=valid_actions)
        
        current_state[team] = state
        current_action[team] = (action, logprob, value)
        
        return jsonify({"action": int(action)})
    
    except Exception as e:
        print(f"Error in select_action: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e), "action": 0}), 500


@app.route('/store_reward', methods=['POST'])
def store_reward():
    """Store reward for the last action taken"""
    try:
        data = request.json
        team = data.get('team', 'team2')
        reward = data.get('reward', 0.0)
        done = data.get('done', False)
        
        agent = team1_agent if team == "team1" else team2_agent
        
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
    """Update the agent (train on stored experiences)"""
    try:
        data = request.json
        team = data.get('team', 'team2')
        
        agent = team1_agent if team == "team1" else team2_agent
        agent.update()
        
        # MEMORY FIX: Force garbage collection after update
        gc.collect()
        
        return jsonify({"status": "updated"})
    
    except Exception as e:
        print(f"Error in update: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return jsonify({"error": str(e)}), 500


@app.route('/save_model', methods=['POST'])
def save_model():
    """Save the model to a specific path"""
    try:
        data = request.json
        team = data.get('team', 'team2')
        path = data.get('path', None)
        
        agent = team1_agent if team == "team1" else team2_agent
        agent.save_model(path)
        
        return jsonify({"status": "saved"})
    
    except Exception as e:
        print(f"Error in save_model: {e}", file=sys.stderr)
        return jsonify({"error": str(e)}), 500


# ========== START SERVER ==========

if __name__ == '__main__':
    print("\n" + "=" * 80, file=sys.stderr)
    print("🚀 MEMORY-OPTIMIZED PPO SERVER READY", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("Endpoints:", file=sys.stderr)
    print("  POST /select_action - Get action from agent", file=sys.stderr)
    print("  POST /store_reward  - Store experience", file=sys.stderr)
    print("  POST /update        - Train agent", file=sys.stderr)
    print("  POST /save_model    - Save model checkpoint", file=sys.stderr)
    print("  GET  /health        - Health check", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print(f"Device: {DEVICE}", file=sys.stderr)
    print(f"State Dimension: {STATE_DIM}", file=sys.stderr)
    print(f"Max Buffer Size: 1000 per agent (MEMORY OPTIMIZED)", file=sys.stderr)
    print("Server starting on http://0.0.0.0:5000", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    
    # Use threaded mode for better performance
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)