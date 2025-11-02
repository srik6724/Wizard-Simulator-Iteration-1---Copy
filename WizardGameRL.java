import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.io.*;

// Mock classes to replace dependencies from original code
class MockWizard {
    private String name;
    private String identity;
    private MockStatsInformation stats;
    private int pips;
    private int shadowGauge;
    private List<String> positiveCharms;
    private List<String> negativeCharms;
    private List<String> shieldsList;
    private List<String> trapsList;
    
    public MockWizard(String name, String identity) {
        this.name = name;
        this.identity = identity;
        this.stats = new MockStatsInformation(3000); // 3000 health
        this.pips = 3;
        this.shadowGauge = 0;
        this.positiveCharms = new ArrayList<>();
        this.negativeCharms = new ArrayList<>();
        this.shieldsList = new ArrayList<>();
        this.trapsList = new ArrayList<>();
    }
    
    // Getters and setters
    public String getName() { return name; }
    public String getIdentity() { return identity; }
    public MockStatsInformation getStatsInformation() { return stats; }
    public int getPips() { return pips; }
    public void setPips(int pips) { this.pips = pips; }
    public int getShadowGauge() { return shadowGauge; }
    public void setShadowGauge(int shadowGauge) { this.shadowGauge = shadowGauge; }
    public List<String> getPositiveCharms() { return positiveCharms; }
    public List<String> getNegativeCharms() { return negativeCharms; }
    public List<String> getShieldsList() { return shieldsList; }
    public List<String> getTrapsList() { return trapsList; }
}

class MockStatsInformation {
    private int health;
    
    public MockStatsInformation(int health) {
        this.health = health;
    }
    
    public int getHealth() { return health; }
    public void setHealth(int health) { this.health = health; }
}

// Main RL Environment
public class WizardGameRL {
    
    // State representation for RL agent
    public static class GameState {
        public final double[] features;
        public final boolean isTerminal;
        public final int currentPlayer;
        
        public GameState(double[] features, boolean isTerminal, int currentPlayer) {
            this.features = features.clone();
            this.isTerminal = isTerminal;
            this.currentPlayer = currentPlayer;
        }
        
        public GameState copy() {
            return new GameState(features, isTerminal, currentPlayer);
        }
    }
    
    // Action representation
    public static class Action {
        public final int type; // 0=pass, 1=cast_spell, 2=discard
        public final int spellIndex; // -1 if not applicable
        public final List<Integer> discardIndices; // empty if not applicable
        
        public Action(int type, int spellIndex, List<Integer> discardIndices) {
            this.type = type;
            this.spellIndex = spellIndex;
            this.discardIndices = new ArrayList<>(discardIndices);
        }
        
        public static Action pass() {
            return new Action(0, -1, Collections.emptyList());
        }
        
        public static Action castSpell(int spellIndex) {
            return new Action(1, spellIndex, Collections.emptyList());
        }
        
        public static Action discard(List<Integer> indices) {
            return new Action(2, -1, indices);
        }
    }
    
    // RL Agent interface
    public interface RLAgent {
        Action selectAction(GameState state, List<Action> validActions);
        void updatePolicy(GameState state, Action action, double reward, GameState nextState, boolean done);
        void saveModel(String filename);
        void loadModel(String filename);
    }
    
    // Simple Q-Learning Agent
    public static class QLearningAgent implements RLAgent {
        private final Map<String, Map<String, Double>> qTable = new HashMap<>();
        private final double learningRate = 0.1;
        private final double discountFactor = 0.95;
        private double epsilon = 0.1; // exploration rate
        private final Random random = new Random();
        
        private String stateToString(GameState state) {
            // Simplified state representation - you'd want more sophisticated encoding
            StringBuilder sb = new StringBuilder();
            for (double f : state.features) {
                sb.append(String.format("%.2f,", f));
            }
            return sb.toString();
        }
        
        private String actionToString(Action action) {
            return String.format("%d_%d_%s", action.type, action.spellIndex, 
                               action.discardIndices.stream().map(String::valueOf).collect(Collectors.joining(",")));
        }
        
        @Override
        public Action selectAction(GameState state, List<Action> validActions) {
            String stateKey = stateToString(state);
            
            // Epsilon-greedy selection
            if (random.nextDouble() < epsilon || !qTable.containsKey(stateKey)) {
                return validActions.get(random.nextInt(validActions.size()));
            }
            
            Map<String, Double> stateActions = qTable.get(stateKey);
            Action bestAction = null;
            double bestValue = Double.NEGATIVE_INFINITY;
            
            for (Action action : validActions) {
                String actionKey = actionToString(action);
                double value = stateActions.getOrDefault(actionKey, 0.0);
                if (value > bestValue) {
                    bestValue = value;
                    bestAction = action;
                }
            }
            
            return bestAction != null ? bestAction : validActions.get(0);
        }
        
        @Override
        public void updatePolicy(GameState state, Action action, double reward, GameState nextState, boolean done) {
            String stateKey = stateToString(state);
            String actionKey = actionToString(action);
            
            qTable.computeIfAbsent(stateKey, k -> new HashMap<>());
            
            double currentQ = qTable.get(stateKey).getOrDefault(actionKey, 0.0);
            double maxNextQ = 0.0;
            
            if (!done && nextState != null) {
                String nextStateKey = stateToString(nextState);
                if (qTable.containsKey(nextStateKey)) {
                    maxNextQ = qTable.get(nextStateKey).values().stream()
                                   .mapToDouble(Double::doubleValue).max().orElse(0.0);
                }
            }
            
            double newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
            qTable.get(stateKey).put(actionKey, newQ);
        }
        
        public void decayEpsilon(double factor) {
            epsilon = Math.max(0.01, epsilon * factor);
        }
        
        @Override
        public void saveModel(String filename) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
                oos.writeObject(qTable);
                oos.writeDouble(epsilon);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        
        @Override
        @SuppressWarnings("unchecked")
        public void loadModel(String filename) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
                Map<String, Map<String, Double>> loadedTable = (Map<String, Map<String, Double>>) ois.readObject();
                qTable.clear();
                qTable.putAll(loadedTable);
                epsilon = ois.readDouble();
            } catch (IOException | ClassNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
    
    // Game Environment (refactored from original Match class)
    public static class GameEnvironment {
        private List<MockWizard> team1 = new ArrayList<>();
        private List<MockWizard> team2 = new ArrayList<>();
        private Random random = new Random();
        private boolean gameOver = false;
        private int winner = -1; // 0 = team1, 1 = team2, -1 = ongoing
        
        public GameEnvironment() {
            initializeWizards();
        }
        
        // State features (you'd expand this significantly)
        private static final int STATE_SIZE = 50;
        
        private void initializeWizards() {
            // Create simple test wizards - in real implementation you'd load from files
            team1.add(createTestWizard("Player1", "balance"));
            team2.add(createTestWizard("Player2", "balance"));
        }
        
        private MockWizard createTestWizard(String name, String school) {
            // Simplified wizard creation - you'd use your existing Wizard/StatsInformation classes
            // For now, creating minimal mock objects to prevent crashes
            return new MockWizard(name, school);
        }
        
        public GameState getState(int playerPerspective) {
            // Check if teams are properly initialized
            if (team1.isEmpty() || team2.isEmpty()) {
                // Return empty state if teams not initialized
                return new GameState(new double[STATE_SIZE], true, playerPerspective);
            }
            
            double[] features = new double[STATE_SIZE];
            int idx = 0;
            
            MockWizard myWizard = (playerPerspective == 0) ? team1.get(0) : team2.get(0);
            MockWizard opponentWizard = (playerPerspective == 0) ? team2.get(0) : team1.get(0);
            
            // Health ratios
            features[idx++] = myWizard.getStatsInformation().getHealth() / 5000.0; // normalize
            features[idx++] = opponentWizard.getStatsInformation().getHealth() / 5000.0;
            
            // Pip counts
            features[idx++] = myWizard.getPips() / 14.0;
            features[idx++] = opponentWizard.getPips() / 14.0;
            
            // Shadow gauge
            features[idx++] = myWizard.getShadowGauge() / 200.0;
            features[idx++] = opponentWizard.getShadowGauge() / 200.0;
            
            // Charm counts (simplified)
            features[idx++] = Math.min(myWizard.getPositiveCharms().size() / 5.0, 1.0);
            features[idx++] = Math.min(myWizard.getNegativeCharms().size() / 5.0, 1.0);
            features[idx++] = Math.min(opponentWizard.getPositiveCharms().size() / 5.0, 1.0);
            features[idx++] = Math.min(opponentWizard.getNegativeCharms().size() / 5.0, 1.0);
            
            // Shield/trap counts
            features[idx++] = Math.min(myWizard.getShieldsList().size() / 5.0, 1.0);
            features[idx++] = Math.min(opponentWizard.getTrapsList().size() / 5.0, 1.0);
            
            // Hand composition (you'd need to track active hands)
            // This is simplified - you'd encode spell types, pip costs, etc.
            for (int i = idx; i < STATE_SIZE; i++) {
                features[i] = random.nextGaussian() * 0.1; // placeholder
            }
            
            return new GameState(features, gameOver, playerPerspective);
        }
        
        public List<Action> getValidActions(int player) {
            List<Action> actions = new ArrayList<>();
            
            // Always can pass
            actions.add(Action.pass());
            
            // Check if teams are properly initialized
            if (team1.isEmpty() || team2.isEmpty()) {
                return actions; // Only return pass action if teams not initialized
            }
            
            // Get valid spell actions (simplified)
            MockWizard wizard = (player == 0) ? team1.get(0) : team2.get(0);
            
            // This would be based on actual hand state - simplified here
            for (int i = 0; i < 7; i++) {
                actions.add(Action.castSpell(i));
            }
            
            // Discard actions (simplified)
            for (int i = 1; i <= 3; i++) {
                List<Integer> discards = new ArrayList<>();
                for (int j = 0; j < i; j++) {
                    discards.add(j);
                }
                actions.add(Action.discard(discards));
            }
            
            return actions;
        }
        
        public double executeAction(int player, Action action) {
            // This would contain the actual game logic from your original code
            // Returns immediate reward
            
            double reward = 0.0;
            
            switch (action.type) {
                case 0: // pass
                    reward = -0.1; // small penalty for passing
                    break;
                case 1: // cast spell
                    reward = executeSpellCast(player, action.spellIndex);
                    break;
                case 2: // discard
                    reward = executeDiscard(player, action.discardIndices);
                    break;
            }
            
            checkGameEnd();
            
            if (gameOver) {
                if (winner == player) {
                    reward += 100.0; // big reward for winning
                } else {
                    reward -= 100.0; // big penalty for losing
                }
            }
            
            return reward;
        }
        
        private double executeSpellCast(int player, int spellIndex) {
            // Check if teams are properly initialized
            if (team1.isEmpty() || team2.isEmpty()) {
                return 0.0;
            }
            
            // Simplified spell casting logic
            MockWizard caster = (player == 0) ? team1.get(0) : team2.get(0);
            MockWizard target = (player == 0) ? team2.get(0) : team1.get(0);
            
            // This would integrate with your existing operateSpellConditions method
            // For now, just simulate some damage
            int damage = random.nextInt(500) + 200;
            int currentHealth = target.getStatsInformation().getHealth();
            int newHealth = Math.max(0, currentHealth - damage);
            target.getStatsInformation().setHealth(newHealth);
            
            // Reward based on damage dealt
            return damage / 1000.0;
        }
        
        private double executeDiscard(int player, List<Integer> indices) {
            // Small reward for strategic discarding
            return indices.size() * 0.05;
        }
        
        private void checkGameEnd() {
            if (!team1.isEmpty() && !team2.isEmpty()) {
                if (team1.get(0).getStatsInformation().getHealth() <= 0) {
                    gameOver = true;
                    winner = 1;
                } else if (team2.get(0).getStatsInformation().getHealth() <= 0) {
                    gameOver = true;
                    winner = 0;
                }
            }
        }
        
        public void reset() {
            gameOver = false;
            winner = -1;
            // Reset wizard states
            if (!team1.isEmpty() && !team2.isEmpty()) {
                team1.get(0).getStatsInformation().setHealth(3000);
                team1.get(0).setPips(3);
                team1.get(0).setShadowGauge(0);
                team1.get(0).getPositiveCharms().clear();
                team1.get(0).getNegativeCharms().clear();
                team1.get(0).getShieldsList().clear();
                team1.get(0).getTrapsList().clear();
                
                team2.get(0).getStatsInformation().setHealth(3000);
                team2.get(0).setPips(5);
                team2.get(0).setShadowGauge(0);
                team2.get(0).getPositiveCharms().clear();
                team2.get(0).getNegativeCharms().clear();
                team2.get(0).getShieldsList().clear();
                team2.get(0).getTrapsList().clear();
            }
        }
        
        public boolean isGameOver() {
            return gameOver;
        }
        
        public int getWinner() {
            return winner;
        }
    }
    
    // Training loop
    public static class Trainer {
        private final GameEnvironment env;
        private final RLAgent agent1;
        private final RLAgent agent2;
        
        public Trainer(GameEnvironment env, RLAgent agent1, RLAgent agent2) {
            this.env = env;
            this.agent1 = agent1;
            this.agent2 = agent2;
        }
        
        public void train(int episodes) {
            for (int episode = 0; episode < episodes; episode++) {
                env.reset();
                
                GameState state1 = env.getState(0);
                GameState state2 = env.getState(1);
                
                int currentPlayer = 0;
                
                while (!env.isGameOver()) {
                    RLAgent currentAgent = (currentPlayer == 0) ? agent1 : agent2;
                    GameState currentState = (currentPlayer == 0) ? state1 : state2;
                    
                    List<Action> validActions = env.getValidActions(currentPlayer);
                    Action selectedAction = currentAgent.selectAction(currentState, validActions);
                    
                    double reward = env.executeAction(currentPlayer, selectedAction);
                    
                    GameState nextState1 = env.getState(0);
                    GameState nextState2 = env.getState(1);
                    GameState nextState = (currentPlayer == 0) ? nextState1 : nextState2;
                    
                    currentAgent.updatePolicy(currentState, selectedAction, reward, nextState, env.isGameOver());
                    
                    state1 = nextState1;
                    state2 = nextState2;
                    currentPlayer = 1 - currentPlayer; // alternate players
                }
                
                // Decay exploration
                if (agent1 instanceof QLearningAgent) {
                    ((QLearningAgent) agent1).decayEpsilon(0.995);
                }
                if (agent2 instanceof QLearningAgent) {
                    ((QLearningAgent) agent2).decayEpsilon(0.995);
                }
                
                if (episode % 100 == 0) {
                    System.out.printf("Episode %d completed. Winner: %d%n", episode, env.getWinner());
                }
            }
        }
    }
    
    // Main method to run training
    public static void main(String[] args) {
        GameEnvironment env = new GameEnvironment();
        RLAgent agent1 = new QLearningAgent();
        RLAgent agent2 = new QLearningAgent();
        
        Trainer trainer = new Trainer(env, agent1, agent2);
        
        System.out.println("Starting training...");
        trainer.train(10000);
        
        // Save trained models
        agent1.saveModel("agent1_model.dat");
        agent2.saveModel("agent2_model.dat");
        
        System.out.println("Training completed!");
    }
}