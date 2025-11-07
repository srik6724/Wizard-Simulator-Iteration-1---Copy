import java.util.concurrent.*;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/*public class Main {
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("WIZARD101 AI vs AI TRAINING");
        System.out.println("=".repeat(80) + "\n");
        
        // Training configuration
        int numMatches = 10;
        int numThreads = 2;  // Run 4 matches in parallel (adjust based on CPU cores)
        
        System.out.println("Configuration:");
        System.out.println("  Total matches: " + numMatches);
        System.out.println("  Parallel workers: " + numThreads);
        System.out.println("  Expected speedup: " + numThreads + "x faster");
        System.out.println();
        
        // Create workers (one per thread)
        System.out.println("Initializing " + numThreads + " workers...\n");
        List<Match_Test_1_0> workers = new ArrayList<>();
        
        for (int i = 0; i < numThreads; i++) {
            System.out.println("Setting up Worker #" + (i + 1) + "...");
            Match_Test_1_0 worker = new Match_Test_1_0();
            worker.initialize();  // Each worker needs team setup (only once)
            workers.add(worker);
        }
        
        System.out.println("\n✅ All workers initialized!\n");
        
        System.out.println("=".repeat(80));
        System.out.println("🤖 STARTING MULTI-THREADED AI vs AI TRAINING");
        System.out.println("=".repeat(80));
        System.out.println("Number of matches: " + numMatches);
        System.out.println("Parallel workers: " + numThreads);
        System.out.println("Both teams: AI-controlled (self-play)");
        System.out.println("Training mode: ENABLED");
        System.out.println("=".repeat(80) + "\n");
        
        System.out.println("Press Enter to start training...");
        try {
            System.in.read();
        } catch (Exception e) {}
        
        // Thread pool for parallel execution
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        
        // Thread-safe counter for progress tracking
        AtomicInteger completedMatches = new AtomicInteger(0);
        Object progressLock = new Object();
        
        long startTime = System.currentTimeMillis();
        
        // Submit all matches to thread pool
        for (int i = 0; i < numMatches; i++) {
            final int matchNum = i + 1;
            final int workerIndex = i % numThreads;  // Round-robin worker assignment
            
            executor.submit(() -> {
                try {
                    Match_Test_1_0 worker = workers.get(workerIndex);
                    
                    System.out.println("[Worker " + (workerIndex + 1) + "] Starting match " + matchNum);
                    worker.playMatch();
                    
                    int completed = completedMatches.incrementAndGet();
                    
                    // Progress updates every 10 matches (thread-safe)
                    if (completed % 10 == 0) {
                        synchronized (progressLock) {
                            long elapsed = (System.currentTimeMillis() - startTime) / 1000;
                            double avgTime = elapsed / (double) completed;
                            int remaining = (int)(avgTime * (numMatches - completed));
                            
                            System.out.println("\n" + "=".repeat(80));
                            System.out.println("📊 TRAINING PROGRESS");
                            System.out.println("=".repeat(80));
                            System.out.println("Matches completed: " + completed + " / " + numMatches + 
                                             " (" + (100 * completed / numMatches) + "%)");
                            System.out.println("Time elapsed: " + elapsed + "s (" + (elapsed / 60) + " min)");
                            System.out.println("Avg time per match: " + String.format("%.1f", avgTime) + "s");
                            System.out.println("Est. time remaining: " + remaining + "s (" + (remaining / 60) + " min)");
                            System.out.println("Active workers: " + numThreads);
                            System.out.println("=".repeat(80) + "\n");
                        }
                    }
                    
                } catch (Exception e) {
                    System.err.println("[Worker " + (workerIndex + 1) + "] Error in match " + matchNum + ": " + e.getMessage());
                    e.printStackTrace();
                }
            });
        }
        
        // Shutdown executor and wait for all matches to complete
        executor.shutdown();
        try {
            System.out.println("Waiting for all matches to complete...\n");
            executor.awaitTermination(24, TimeUnit.HOURS);
        } catch (InterruptedException e) {
            System.err.println("Training interrupted!");
            e.printStackTrace();
        }
        
        // Training complete
        long totalTime = (System.currentTimeMillis() - startTime) / 1000;
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("✅ TRAINING SESSION COMPLETE!");
        System.out.println("=".repeat(80));
        System.out.println("Total matches: " + numMatches);
        System.out.println("Parallel workers: " + numThreads);
        System.out.println("Total time: " + totalTime + "s (" + (totalTime / 60) + " min " + (totalTime % 60) + "s)");
        System.out.println("Avg time per match: " + String.format("%.2f", totalTime / (double)numMatches) + "s");
        System.out.println("Speedup: ~" + String.format("%.1f", numThreads * 0.8) + "x faster than single-threaded");
        System.out.println("\n📁 Results saved to:");
        System.out.println("   • match_log.txt (all match details)");
        System.out.println("   • wins_log_t1.txt (Team 1 victories)");
        System.out.println("   • wins_log_t2.txt (Team 2 victories)");
        System.out.println("   • ppo_model_team1.pth (trained Team 1 AI)");
        System.out.println("   • ppo_model_team2.pth (trained Team 2 AI)");
        System.out.println("   • checkpoints/ (periodic backups)");
        System.out.println("=".repeat(80));
    }
}*/
public class Main {
    public static void main(String[]args) {
       //Match m = new Match();
       //Match_Test m = new Match_Test();
        //Match_ m = new Match_();

        System.out.println("=".repeat(80));
        System.out.println("WIZARD101 AI vs AI TRAINING");
        System.out.println("=".repeat(80) + "\n");
        
        // Create match object ONCE
        Match_Test_1_0 match = new Match_Test_1_0();
        
        // Initialize teams ONCE (interactive setup)
        match.initialize();
        
        // Training configuration
        int numMatches = 100;  // Change this to train for more/fewer matches
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("🤖 STARTING AI vs AI TRAINING SESSION");
        System.out.println("=".repeat(80));
        System.out.println("Number of matches: " + numMatches);
        System.out.println("Both teams: AI-controlled (self-play)");
        System.out.println("Training mode: ENABLED");
        System.out.println("=".repeat(80) + "\n");
        
        // Pause before starting
        System.out.println("Press Enter to start training...");
        try {
            System.in.read();
        } catch (Exception e) {}
        
        // Training loop
        long startTime = System.currentTimeMillis();
        
        for (int i = 0; i < numMatches; i++) {
            System.out.println("\n" + "█".repeat(80));
            System.out.println("█ MATCH " + (i + 1) + " / " + numMatches);
            System.out.println("█".repeat(80));
            
            match.playMatch();  // Play one match
            
            // Progress updates every 10 matches
            if ((i + 1) % 10 == 0) {
                long elapsed = (System.currentTimeMillis() - startTime) / 1000;
                double avgTime = elapsed / (double)(i + 1);
                int remaining = (int)(avgTime * (numMatches - i - 1));
                
                System.out.println("\n" + "=".repeat(80));
                System.out.println("📊 TRAINING PROGRESS");
                System.out.println("=".repeat(80));
                System.out.println("Matches completed: " + (i + 1) + " / " + numMatches + 
                                 " (" + (100 * (i + 1) / numMatches) + "%)");
                System.out.println("Time elapsed: " + elapsed + "s (" + (elapsed / 60) + " min)");
                System.out.println("Avg time per match: " + String.format("%.1f", avgTime) + "s");
                System.out.println("Est. time remaining: " + remaining + "s (" + (remaining / 60) + " min)");
                System.out.println("=".repeat(80) + "\n");
            }
        }
        
        // Training complete
        long totalTime = (System.currentTimeMillis() - startTime) / 1000;
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("✅ TRAINING SESSION COMPLETE!");
        System.out.println("=".repeat(80));
        System.out.println("Total matches: " + numMatches);
        System.out.println("Total time: " + totalTime + "s (" + (totalTime / 60) + " min " + (totalTime % 60) + "s)");
        System.out.println("Avg time per match: " + String.format("%.2f", totalTime / (double)numMatches) + "s");
        System.out.println("\n📁 Results saved to:");
        System.out.println("   • match_log.txt (all match details)");
        System.out.println("   • wins_log_t1.txt (Team 1 victories)");
        System.out.println("   • wins_log_t2.txt (Team 2 victories)");
        System.out.println("   • ppo_model_team1.pth (trained Team 1 AI)");
        System.out.println("   • ppo_model_team2.pth (trained Team 2 AI)");
        System.out.println("   • checkpoints/ (periodic backups)");
        System.out.println("=".repeat(80));
    }
}