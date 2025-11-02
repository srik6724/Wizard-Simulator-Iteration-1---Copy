import java.util.*;

public class ModelUsageExample {
    
    // 1. Load a trained model and play against it
    public static void playAgainstTrainedAgent() {
        WizardGameRL.GameEnvironment env = new WizardGameRL.GameEnvironment();
        WizardGameRL.QLearningAgent trainedAgent = new WizardGameRL.QLearningAgent();
        
        // Load the trained model
        trainedAgent.loadModel("agent1_model.dat");
        
        // Create a human-playable interface
        Scanner scanner = new Scanner(System.in);
        env.reset();
        
        int currentPlayer = 0; // 0 = human, 1 = AI
        
        while (!env.isGameOver()) {
            WizardGameRL.GameState currentState = env.getState(currentPlayer);
            
            if (currentPlayer == 0) {
                // Human turn
                System.out.println("Your turn! Available actions:");
                List<WizardGameRL.Action> actions = env.getValidActions(currentPlayer);
                
                for (int i = 0; i < actions.size(); i++) {
                    WizardGameRL.Action action = actions.get(i);
                    System.out.printf("%d: ", i);
                    switch (action.type) {
                        case 0: System.out.println("Pass"); break;
                        case 1: System.out.println("Cast spell " + action.spellIndex); break;
                        case 2: System.out.println("Discard cards " + action.discardIndices); break;
                    }
                }
                
                System.out.print("Choose action (0-" + (actions.size() - 1) + "): ");
                int choice = scanner.nextInt();
                
                if (choice >= 0 && choice < actions.size()) {
                    double reward = env.executeAction(currentPlayer, actions.get(choice));
                    System.out.println("Action executed. Reward: " + reward);
                }
                
            } else {
                // AI turn
                System.out.println("AI is thinking...");
                List<WizardGameRL.Action> actions = env.getValidActions(currentPlayer);
                WizardGameRL.Action aiAction = trainedAgent.selectAction(currentState, actions);
                
                double reward = env.executeAction(currentPlayer, aiAction);
                
                System.out.print("AI chose: ");
                switch (aiAction.type) {
                    case 0: System.out.println("Pass"); break;
                    case 1: System.out.println("Cast spell " + aiAction.spellIndex); break;
                    case 2: System.out.println("Discard cards " + aiAction.discardIndices); break;
                }
                System.out.println("AI reward: " + reward);
            }
            
            currentPlayer = 1 - currentPlayer; // Switch players
        }
        
        int winner = env.getWinner();
        if (winner == 0) {
            System.out.println("You won!");
        } else {
            System.out.println("AI won!");
        }
        
        scanner.close();
    }
    
    // 2. Evaluate trained models against each other
    public static void evaluateModels() {
        WizardGameRL.QLearningAgent agent1 = new WizardGameRL.QLearningAgent();
        WizardGameRL.QLearningAgent agent2 = new WizardGameRL.QLearningAgent();
        
        // Load both models
        agent1.loadModel("agent1_model.dat");
        agent2.loadModel("agent2_model.dat");
        
        // Play evaluation games
        int gamesCount = 100;
        int agent1Wins = 0;
        int agent2Wins = 0;
        
        for (int game = 0; game < gamesCount; game++) {
            WizardGameRL.GameEnvironment env = new WizardGameRL.GameEnvironment();
            env.reset();
            
            int currentPlayer = 0;
            
            while (!env.isGameOver()) {
                WizardGameRL.GameState state = env.getState(currentPlayer);
                List<WizardGameRL.Action> actions = env.getValidActions(currentPlayer);
                
                WizardGameRL.RLAgent currentAgent = (currentPlayer == 0) ? agent1 : agent2;
                WizardGameRL.Action action = currentAgent.selectAction(state, actions);
                
                env.executeAction(currentPlayer, action);
                currentPlayer = 1 - currentPlayer;
            }
            
            int winner = env.getWinner();
            if (winner == 0) agent1Wins++;
            else if (winner == 1) agent2Wins++;
        }
        
        System.out.printf("Evaluation Results over %d games:%n", gamesCount);
        System.out.printf("Agent 1 wins: %d (%.1f%%)%n", agent1Wins, (agent1Wins * 100.0) / gamesCount);
        System.out.printf("Agent 2 wins: %d (%.1f%%)%n", agent2Wins, (agent2Wins * 100.0) / gamesCount);
    }
    
    // 3. Continue training from saved models
    public static void continueTraining() {
        WizardGameRL.GameEnvironment env = new WizardGameRL.GameEnvironment();
        WizardGameRL.QLearningAgent agent1 = new WizardGameRL.QLearningAgent();
        WizardGameRL.QLearningAgent agent2 = new WizardGameRL.QLearningAgent();
        
        // Load existing models
        agent1.loadModel("agent1_model.dat");
        agent2.loadModel("agent2_model.dat");
        
        // Continue training
        WizardGameRL.Trainer trainer = new WizardGameRL.Trainer(env, agent1, agent2);
        
        System.out.println("Continuing training from saved models...");
        trainer.train(5000); // Train for 5000 more episodes
        
        // Save updated models
        agent1.saveModel("agent1_model_extended.dat");
        agent2.saveModel("agent2_model_extended.dat");
        
        System.out.println("Extended training completed!");
    }
    
    // 4. Analyze model behavior
    public static void analyzeModel() {
        WizardGameRL.QLearningAgent agent = new WizardGameRL.QLearningAgent();
        agent.loadModel("agent1_model.dat");
        
        WizardGameRL.GameEnvironment env = new WizardGameRL.GameEnvironment();
        env.reset();
        
        // Test specific game states
        System.out.println("Analyzing agent behavior in different states:");
        
        for (int scenario = 0; scenario < 5; scenario++) {
            env.reset();
            WizardGameRL.GameState state = env.getState(0);
            List<WizardGameRL.Action> actions = env.getValidActions(0);
            
            System.out.println("\nScenario " + (scenario + 1) + ":");
            System.out.println("State features (first 10): ");
            for (int i = 0; i < Math.min(10, state.features.length); i++) {
                System.out.printf("%.3f ", state.features[i]);
            }
            System.out.println();
            
            WizardGameRL.Action chosenAction = agent.selectAction(state, actions);
            System.out.print("Agent chose: ");
            switch (chosenAction.type) {
                case 0: System.out.println("Pass"); break;
                case 1: System.out.println("Cast spell " + chosenAction.spellIndex); break;
                case 2: System.out.println("Discard cards " + chosenAction.discardIndices); break;
            }
        }
    }
    
    // 5. Export strategy insights
    public static void exportStrategyInsights() {
        WizardGameRL.QLearningAgent agent = new WizardGameRL.QLearningAgent();
        agent.loadModel("agent1_model.dat");
        
        // Access the Q-table (you'd need to make it public or add a getter)
        System.out.println("Strategy insights would require exposing the Q-table");
        System.out.println("You could add methods to the QLearningAgent to:");
        System.out.println("- Export top-valued state-action pairs");
        System.out.println("- Show most frequently visited states");
        System.out.println("- Display action preferences in different game phases");
    }
    
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        System.out.println("Choose an option:");
        System.out.println("1. Play against trained AI");
        System.out.println("2. Evaluate models against each other");
        System.out.println("3. Continue training from saved models");
        System.out.println("4. Analyze model behavior");
        System.out.println("5. Show strategy insights info");
        
        int choice = scanner.nextInt();
        
        switch (choice) {
            case 1:
                playAgainstTrainedAgent();
                break;
            case 2:
                evaluateModels();
                break;
            case 3:
                continueTraining();
                break;
            case 4:
                analyzeModel();
                break;
            case 5:
                exportStrategyInsights();
                break;
            default:
                System.out.println("Invalid choice");
        }
        
        scanner.close();
    }
}
