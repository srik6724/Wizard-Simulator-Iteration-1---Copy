import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import java.io.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Match_Test_1_1 {
    private final Scanner sc = new Scanner(System.in);
    private List<Wizard> t1Wizards = new ArrayList<>();
    private List<Wizard> t2Wizards = new ArrayList<>();
    private int initial = 1; 
    private int initialt2 = 1;
    private int number = 7; 
    private int numbert2 = 7;
    private boolean firstIteration = true;
    private boolean firstIterationTeam2 = true;
    private boolean firstPowerPipIterationt2 = true;
    List<Spell> activeHand = new ArrayList<>(); 
    Set<Integer> chosen = new HashSet<>();
    List<Spell> activeHandTeam2 = new ArrayList<>();
    Set<Integer> chosenTeam2 = new HashSet<>();
    private Bubble bubble; 
    private boolean winner = false;
    
    // Wizard references (stored after initialization)
    private Wizard w;
    private Wizard wt2;
    private boolean isInitialized = false;
    
    // PPO Agent integration
    private boolean usePPOForTeam1 = true;   // Team 1 uses AI (self-play!)
    private boolean usePPOForTeam2 = true;   // Team 2 uses AI by default
    private String pythonPath = "python3";   // Adjust if needed
    private String ppoScriptPath = "ppo_agent.py";
    private int roundNumber = 0;
    private boolean trainingMode = true;     // Set to false for evaluation
    private double lastRewardTeam1 = 0.0;
    private double lastRewardTeam2 = 0.0;
    
    // Match logging for analysis
    private boolean logMatches = true;       // Enable match logging
    private String logFilePath = "match_log.txt";
    private String winsLogPath = "wins_log.txt";
    private StringBuilder matchLog = new StringBuilder();
    private static int matchNumber = 0;

    /**
     * Empty constructor - doesn't do anything
     */
    public Match_Test_1_1() {
        // Empty - call initialize() to setup teams
    }
    
    /**
     * Initialize teams (call this ONCE before training)
     */
    public void initialize() {
        if (isInitialized) {
            System.out.println("⚠ Already initialized!");
            return;
        }
        
        System.out.println("\n" + "=".repeat(80));
        System.out.println("TEAM SETUP");
        System.out.println("=".repeat(80) + "\n");
        
        for(int i = 0; i < 2; i++) {
            System.out.println("Enter size of your team.");
            int size = sc.nextInt();
            sc.nextLine();
            Scanner test = new Scanner(System.in);
            for(int y = 0;  y < size; y++) {
                System.out.println("Enter name of your team member.");
                String name = test.nextLine();
                System.out.println("Enter name of your identity associated with your team member");
                Scanner newScanner = new Scanner(System.in);
                String identity = newScanner.nextLine();
                StatsInformation information = null;
                String filePath = "stats_t" + (i+1) + "_" + (y+1) + ".txt";
                try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
                    int health = Integer.parseInt(br.readLine());
                    String[] fields_1 = br.readLine().split(",");
                    String[] schools = {"fire", "ice", "storm", "myth", "life", "death", "balance", "shadow"};
                    Map<String, Integer> damage = new HashMap<>();
                    Map<String, Integer> resist = new HashMap<>();
                    Map<String, Integer> accuracy = new HashMap<>();
                    Map<String, Integer> critical = new HashMap<>();
                    Map<String, Integer> block = new HashMap<>();
                    Map<String, Integer> pierce = new HashMap<>();
                    Map<String, Integer> pipConversion = new HashMap<>();

                    for(int z = 0; z < fields_1.length; z++) damage.put(schools[z], Integer.parseInt(fields_1[z]));
                    String[] fields_2 = br.readLine().split(",");
                    for(int z = 0; z < fields_2.length; z++) resist.put(schools[z], Integer.parseInt(fields_2[z]));
                    String[] fields_3 = br.readLine().split(",");
                    for(int z = 0; z < fields_3.length; z++) accuracy.put(schools[z], Integer.parseInt(fields_3[z]));
                    String[] fields_4 = br.readLine().split(",");
                    for(int z = 0; z < fields_4.length; z++) critical.put(schools[z], Integer.parseInt(fields_4[z]));
                    String[] fields_5 = br.readLine().split(",");
                    for(int z = 0; z < fields_5.length; z++) block.put(schools[z], Integer.parseInt(fields_5[z]));
                    String[] fields_6 = br.readLine().split(",");
                    for(int z = 0; z < fields_6.length; z++) pierce.put(schools[z], Integer.parseInt(fields_6[z]));
                    int stunResist = Integer.parseInt(br.readLine());
                    int incoming = Integer.parseInt(br.readLine());
                    int outgoing = Integer.parseInt(br.readLine());
                    String[] fields_7 = br.readLine().split(",");
                    for(int z = 0; z < fields_7.length; z++) pipConversion.put(schools[z], Integer.parseInt(fields_7[z]));
                    int powerPip = Integer.parseInt(br.readLine());
                    int shadowPipRating = Integer.parseInt(br.readLine());
                    int archMasteryRating = Integer.parseInt(br.readLine());
                    information = new StatsInformation(
                            health, damage, resist, accuracy, critical, block, pierce,
                            stunResist, incoming, outgoing, pipConversion, powerPip, shadowPipRating, archMasteryRating
                    );
                    System.out.println(information.toString());
                } catch (IOException e) {
                    e.printStackTrace();
                }

                List<Spell> mainDeckSpells = new ArrayList<>();
                String filePath2 = "deck_main_t" + (i+1) + "_" + (y+1) + ".txt";
                int a = 1;
                try(BufferedReader br = new BufferedReader(new FileReader(filePath2)))  {
                    if(a == 1) { br.readLine(); a = 0; }
                    String useLine;
                    while((useLine=br.readLine()) != null) {
                        String[] fields = useLine.split(",");
                        if(fields[1].equals("X")) {
                            Spell spell = new Spell(fields[0], 0, 14, Integer.parseInt(fields[2]), fields[3], "");
                            mainDeckSpells.add(spell);
                            continue;
                        }
                        String nameOfSpell = fields[0];
                        int pips = Integer.parseInt(fields[1]);
                        int pipChance = Integer.parseInt(fields[2]);
                        String school = fields[3];
                        Spell spell = new Spell(nameOfSpell, pips, pipChance, school, "");
                        mainDeckSpells.add(spell);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                Deck mainDeck = new Deck(mainDeckSpells);

                int b = 1;
                List<Spell> tcSpells = new ArrayList<>();
                String filePath3 = "deck_tc_t" + (i+1) + "_" + (y+1) + ".txt";
                try(BufferedReader br = new BufferedReader(new FileReader(filePath3)))  {
                    if(b == 1) { br.readLine(); b = 0; }
                    String useLine;
                    while((useLine=br.readLine()) != null) {
                        String[] fields = useLine.split(",");
                        String nameOfSpell = fields[0];
                        int pips = Integer.parseInt(fields[1]);
                        int pipChance = Integer.parseInt(fields[2]);
                        String school = fields[3];
                        Spell spell = new Spell(nameOfSpell, pips, pipChance, school, "");
                        tcSpells.add(spell);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
                Deck tcDeck = new Deck(tcSpells);

                int pips = 0; 
                if((i+1) == 1) {
                    pips = 3;
                }
                else if((i+1) == 2) {
                    pips = 5; 
                }

                Wizard wizard = new Wizard(name, identity, information, mainDeck, tcDeck, pips);
                if((i+1) == 1) {
                    t1Wizards.add(wizard);
                    w = wizard;  // Store Team 1 wizard
                } else {
                    t2Wizards.add(wizard);
                    wt2 = wizard;  // Store Team 2 wizard
                }
            }
        }
        
        printWizardTeamInformation();
        
        isInitialized = true;
        System.out.println("\n✅ Teams initialized successfully!\n");
    }
    
    /**
     * Play a single match (can be called repeatedly)
     */
    public void playMatch() {
        if (!isInitialized) {
            throw new IllegalStateException("❌ Must call initialize() before playMatch()!");
        }
        
        matchNumber++;
        System.out.println("\n" + "=".repeat(80));
        System.out.println("MATCH #" + matchNumber);
        System.out.println("=".repeat(80) + "\n");
        
        // Reset game state for new match
        resetGameState();
        
        // Initialize match logging
        initializeMatchLog(w, wt2);
        
        // Reset winner flag
        winner = false;
        roundNumber = 0;
        
        // Play until someone wins
        while(!winner) {
            roundNumber++;
            System.out.println("\n--- ROUND " + roundNumber + " ---");
            startRound();
        }
        
        // Match over - agents already updated in startRound()
        System.out.println("=".repeat(80) + "\n");
    }
    
    /**
     * Reset game state between matches
     */
    private void resetGameState() {
        System.out.println("Resetting game state for new match...\n");
        
        // Reset health
        if (w != null) w.getStatsInformation().setHealth(5000);
        if (wt2 != null) wt2.getStatsInformation().setHealth(5000);
        
        // Reset pips
        if (w != null) w.setPips(3);
        if (wt2 != null) wt2.setPips(5);
        
        // Reset shadow gauge
        if (w != null) w.setShadowGauge(0);
        if (wt2 != null) wt2.setShadowGauge(0);
        
        // Clear hands
        activeHand.clear();
        activeHandTeam2.clear();
        
        // Reset chosen cards
        chosen.clear();
        chosenTeam2.clear();
        
        // Reset hand sizes
        number = 7;
        numbert2 = 7;
        
        // Reset iteration flags
        firstIteration = true;
        firstIterationTeam2 = true;
        firstPowerPipIterationt2 = true;
        initial = 1;
        initialt2 = 1;
        
        // Clear auras and effects
        if (w != null) {
            w.setAura(null);
            w.setInfallible(null);
            w.getPositiveCharms().clear();
            w.getNegativeCharms().clear();
            w.getShieldsList().clear();
            w.getTrapsList().clear();
            w.getEffects().clear();
            w.getOvertimes().clear();
            if (w.getBlades() != null) w.getBlades().clear();
            if (w.getWeaknesses() != null) w.getWeaknesses().clear();
            if (w.getShields() != null) w.getShields().clear();
            if (w.getTraps() != null) w.getTraps().clear();
        }
        
        if (wt2 != null) {
            wt2.setAura(null);
            wt2.setInfallible(null);
            wt2.getPositiveCharms().clear();
            wt2.getNegativeCharms().clear();
            wt2.getShieldsList().clear();
            wt2.getTrapsList().clear();
            wt2.getEffects().clear();
            wt2.getOvertimes().clear();
            if (wt2.getBlades() != null) wt2.getBlades().clear();
            if (wt2.getWeaknesses() != null) wt2.getWeaknesses().clear();
            if (wt2.getShields() != null) wt2.getShields().clear();
            if (wt2.getTraps() != null) wt2.getTraps().clear();
        }
        
        // Reset decks (restore all cards)
        if (w != null) {
            List<Spell> mainSpells = w.getMainDeck().getSpells();
            List<String> mainNames = w.getMainDeck().getSpellNames();
            for (int i = 0; i < mainSpells.size() && i < mainNames.size(); i++) {
                mainSpells.get(i).setName(mainNames.get(i));
            }
            
            List<Spell> tcSpells = w.getTcDeck().getSpells();
            List<String> tcNames = w.getTcDeck().getSpellNames();
            for (int i = 0; i < tcSpells.size() && i < tcNames.size(); i++) {
                tcSpells.get(i).setName(tcNames.get(i));
            }
        }
        
        if (wt2 != null) {
            List<Spell> mainSpells = wt2.getMainDeck().getSpells();
            List<String> mainNames = wt2.getMainDeck().getSpellNames();
            for (int i = 0; i < mainSpells.size() && i < mainNames.size(); i++) {
                mainSpells.get(i).setName(mainNames.get(i));
            }
            
            List<Spell> tcSpells = wt2.getTcDeck().getSpells();
            List<String> tcNames = wt2.getTcDeck().getSpellNames();
            for (int i = 0; i < tcSpells.size() && i < tcNames.size(); i++) {
                tcSpells.get(i).setName(tcNames.get(i));
            }
        }
        
        // Reset bubble
        bubble = null;
        
        System.out.println("✓ Game state reset complete\n");
    }

    // Marks the selected card as used ("X") in its source deck.
    private void markCardUsedInDecks(Spell selected, List<Spell> mainDeckSpells, List<Spell> tcDeckSpells) {
        int mdIdx = mainDeckSpells.indexOf(selected);
        if (mdIdx >= 0) {
            mainDeckSpells.get(mdIdx).setName("X");
            return;
        }

        String name = selected.getName();
        for (int i = 0; i < tcDeckSpells.size(); i++) {
            Spell s = tcDeckSpells.get(i);
            if (!"X".equals(s.getName()) && s.getName().equals(name)) {
                s.setName("X");
                return;
            }
        }
    }

    private static int clamp(int x, int lo, int hi) { 
        return Math.max(lo, Math.min(hi, x)); 
    }

    private static int parsePercentFromText(String text) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.isDigit(c) || (c == '-' && sb.length() == 0)) sb.append(c);
        }
        if (sb.length() == 0) return 0;
        try { return Math.abs(Integer.parseInt(sb.toString())); } catch (Exception e) { return 0; }
    }

    private int checkForBuffs(int spellDamage, String pierceSchool, String spellName, String school, Wizard w, Wizard wt2) {
        System.out.println("Checking for blades.");
        Set<String> uniqueBlades = new HashSet<>();
        for(String entry: w.getBlades().keySet()) {
            if(entry.equals(school) || entry.equals("universal")) {
                for(String blade: w.getBlades().get(entry)) {
                    uniqueBlades.add(blade);
                }
            }
        }
        
        for(String blade: uniqueBlades) {
            String number = ""; 
            for(int i = 0; i < blade.length(); i++) {
                char c = blade.charAt(i);
                if(Character.isDigit(c)) {
                    number += c;
                }
            }
            int value = Integer.parseInt(number);
            int damageToAdd = (int)((value/100.0) * spellDamage);
            spellDamage += damageToAdd;
        }

        for (String blade : uniqueBlades) {
            Iterator<Map.Entry<String, String>> it = w.getBlades().entries().iterator();
            while (it.hasNext()) {
                Map.Entry<String, String> entry = it.next();
                String key = entry.getKey();
                String value = entry.getValue();

                if (blade.equals(value)) {
                    it.remove();
                    w.getPositiveCharms().remove(value);
                }
            }
        }

        Set<String> uniqueTraps = new HashSet<>(); 
        for(String entry: wt2.getTraps().keySet()) {
            if(entry.equals(school) || entry.equals("universal")) {
                for(String trap: wt2.getTraps().get(entry)) {
                    uniqueTraps.add(trap);
                }
            }
        }

        for(String trap: uniqueTraps) {
            String number = ""; 
            for(int i = 0; i < trap.length(); i++) {
                char c = trap.charAt(i);
                if(Character.isDigit(c)) {
                    number += c;
                }
            }
            int value = Integer.parseInt(number);
            int damageToAdd = (int)((value/100.0) * spellDamage);
            spellDamage += damageToAdd;
        }

        for (String trap : uniqueTraps) {
            Iterator<Map.Entry<String, String>> it = wt2.getTraps().entries().iterator();
            while (it.hasNext()) {
                Map.Entry<String, String> entry = it.next();
                String key = entry.getKey();
                String value = entry.getValue();

                if (trap.equals(value)) {
                    it.remove();
                    wt2.getTrapsList().remove(value);
                }
            }
        }

        Set<String> uniqueWeaknesses = new HashSet<>(); 
        for(String entry: w.getWeaknesses().keySet()) {
            if(entry.equals(school) || entry.equals("universal")) {
                for(String weakness: w.getWeaknesses().get(entry)) {
                    uniqueWeaknesses.add(weakness);
                }
            }
        }

        for(String weakness: uniqueWeaknesses) {
            String number = ""; 
            for(int i = 0; i < weakness.length(); i++) {
                char c = weakness.charAt(i);
                if(Character.isDigit(c)) {
                    number += c;
                }
            }
            int value = Integer.parseInt(number);
            int damageToAdd = (int)((value/100.0) * spellDamage);
            spellDamage -= damageToAdd;
        }

        for (String weakness : uniqueWeaknesses) {
            Iterator<Map.Entry<String, String>> it = w.getWeaknesses().entries().iterator();
            while (it.hasNext()) {
                Map.Entry<String, String> entry = it.next();
                String key = entry.getKey();
                String value = entry.getValue();

                if (weakness.equals(value)) {
                    it.remove();
                    w.getWeaknessesList().remove(value);
                }
            }
        }

        boolean shrikeActive = w.getShadowEffects() != null && w.getShadowEffects().contains("Shrike");
        int basePierce = 0;
        try {
            Integer p = w.getStatsInformation().getPierce().get(pierceSchool);
            basePierce = (p == null ? 0 : p);
        } catch (Exception ignore) {}
        int effectivePierce = clamp(basePierce + (shrikeActive ? 50 : 0), 0, 100);

        Set<String> shieldsToApply = new HashSet<>();
        for (String entry : wt2.getShields().keySet()) {
            if (entry.equals(school) || entry.equals("universal")) {
                for(String shield: wt2.getShields().get(entry)) {
                    shieldsToApply.add(shield);
                }
            }
        }

        double dmg = spellDamage;
        int pierceLeft = effectivePierce;

        for (String shield : shieldsToApply) {
            int shieldPct = parsePercentFromText(shield);
            int usePierce = Math.min(pierceLeft, shieldPct);
            int shieldAfter = clamp(shieldPct - usePierce, 0, 100);
            pierceLeft = clamp(pierceLeft - usePierce, 0, 100);

            if (shieldAfter > 0) {
                dmg = Math.max(0, Math.round(dmg * (1.0 - shieldAfter / 100.0)));
            }

            Iterator<Map.Entry<String, String>> itS = wt2.getShields().entries().iterator();
            while (itS.hasNext()) {
                Map.Entry<String, String> e = itS.next();
                if (shield.equals(e.getValue())) {
                    itS.remove();
                    if (wt2.getShieldsList() != null) wt2.getShieldsList().remove(shield);
                    break;
                }
            }
        }

        if(w.getAura() != null) {
            String description = w.getAura().getDescription();
            String number = ""; 
            for(int i = 0; i < description.length(); i++) {
                char c = description.charAt(i);
                if(Character.isDigit(c)) {
                    number += c;
                }
            }
            int value = Integer.parseInt(number);
            int damageToAdd = (int)((value/100.0) * spellDamage);
            spellDamage += damageToAdd;
        }

        if(bubble != null) {
            String description = bubble.getDescription();
            String number = ""; 
            for(int i = 0; i < description.length(); i++) {
                char c = description.charAt(i);
                if(Character.isDigit(c)) {
                    number += c;
                }
            }
            int value = Integer.parseInt(number);
            int damageToAdd = (int)((value/100.0) * spellDamage);
            if(bubble.getSchool().equals(school)) {
                spellDamage += damageToAdd;
            }
        }
        return spellDamage;
    }

    private void offTurnPipGain(Wizard wiz) {
        int chance = wiz.getStatsInformation().getPowerPip();
        int roll = new Random().nextInt(100);
        String gained;
        int inc;
        if (roll < chance) {
            gained = "2|: 1 power pip";
            inc = 2;
        } else {
            gained = "1|: 1 regular pip";
            inc = 1;
        }

        int next = Math.min(14, wiz.getPips() + inc);
        wiz.setPips(next);

        String prev = wiz.getPipsByCount();
        wiz.setPipsByCount(gained + (prev == null ? "" : prev));
    }

    private int countTCAndMainDeckSpells(Wizard w) {
        int countTCAndMain = 0;
        for(Spell spell: w.getMainDeck().getSpells()) {
            if(!spell.getName().equals("X")) {
                countTCAndMain++;
            }
        }
        for(Spell spell: w.getTcDeck().getSpells()) {
            if(!spell.getName().equals("X")) {
                countTCAndMain++;
            }
        }
        return countTCAndMain;
    }

    public void operateSpellConditions(String nameOfSpell, Spell selected_, Wizard w, Wizard wt2) {
    if(nameOfSpell.equals("glacial fortress")) {
            System.out.println("Checking for any negative charms on self.");
                List<String> negativeCharms = w.getNegativeCharms();
                if(negativeCharms.size() > 0) {
                    System.out.println("There are at least 3 negative charms to remove on the wizard.");
                }
                else {
                    System.out.println("You do not have up to 3 negative charms to remove on the wizard.");
                }
                try {
                    System.out.println("Removing " + negativeCharms.get(0) + "," + negativeCharms.get(1) + negativeCharms.get(2)); 
                    String removed = negativeCharms.remove(0); 
                    if(removed.contains("weakness") && !w.getWeaknessesList().isEmpty()) {
                        w.getWeaknessesList().remove(0);
                        Map.Entry<String, String> first = w.getWeaknesses().entries().iterator().next();
                        w.getWeaknesses().remove(first.getKey(), first.getValue());
                    }
                    else if(removed.contains("infection") && !w.getInfections().isEmpty()) {
                        w.getInfections().remove(0);
                    }
                    w.getShieldsList().add("Shield 1 -25%");
                    w.getShields().put("universal", "Shield -25%");
                    String removed2 = negativeCharms.remove(0);
                    if(removed2.contains("weakness") && !w.getWeaknessesList().isEmpty()) {
                        w.getWeaknessesList().remove(0);
                        Map.Entry<String, String> first = w.getWeaknesses().entries().iterator().next();
                        w.getWeaknesses().remove(first.getKey(), first.getValue());
                    }
                    else if(removed2.contains("infection") && !w.getInfections().isEmpty()) {
                        w.getInfections().remove(0);
                    }
                    w.getShieldsList().add("Shield 2 - 25%");
                    w.getShields().put("universal", "Shield -25%");
                    String removed3 = negativeCharms.remove(0); 
                    if(removed3.contains("weakness") && !w.getWeaknessesList().isEmpty()) {
                        w.getWeaknessesList().remove(0);
                        Map.Entry<String, String> first = w.getWeaknesses().entries().iterator().next();
                        w.getWeaknesses().remove(first.getKey(), first.getValue());
                    }
                    else if(removed3.contains("infection") && !w.getInfections().isEmpty()) {
                        w.getInfections().remove(0);
                    }
                    w.getShieldsList().add("Shield 3 - 25%");
                    w.getShields().put("universal", "Shield -25%");
                } catch (Exception e) {
                    System.out.println("Glacial Fortress spell finished.");
                }
        }
        else if(nameOfSpell.equals("stun block")) {
            System.out.println("Placing 2 stun effects on self.");
            w.getEffects().add("Stun Effect");
            w.getEffects().add("Stun Effect");
        }
        else if(nameOfSpell.equals("jinn's defense")) {
            System.out.println("Select a target. The following targets are " + t2Wizards.get(0).getName());
            String target = sc.nextLine();
            if(target.equals(t2Wizards.get(0).getName())) {
                System.out.println("Looking for shields on target.");
                List<String> shields = t2Wizards.get(0).getShieldsList();
                if(shields.size() == 0) {
                    System.out.println("No shields on target to remove.");
                    return;
                }
                System.out.println("Shields on target:" + shields);
                System.out.println("Removing shields on opponent.");
                try {
                    shields.remove(0);
                    Map.Entry<String, String> first = w.getShields().entries().iterator().next();
                    w.getShields().remove(first.getKey(), first.getValue());
                    w.getShieldsList().add("Shield -25%");
                    w.getShields().put("universal", "-25% shield");
                    shields.remove(0);
                    Map.Entry<String, String> first2 = w.getShields().entries().iterator().next();
                    w.getShields().remove(first2.getKey(), first2.getValue());
                    w.getShieldsList().add("Shield -25%");
                    w.getShields().put("universal", "-25% shield");
                    shields.remove(0);
                    Map.Entry<String, String> first3 = w.getShields().entries().iterator().next();
                    w.getShields().remove(first3.getKey(), first3.getValue());
                    w.getShieldsList().add("Shield -25%");
                    w.getShields().put("universal", "-25% shield");
                } catch (Exception e) {
                    System.out.println("Finished casting jinn's defense.");
                }
            }
        }
        else if(nameOfSpell.equals("magnify")) {
            w.setAura(new Aura("Magnify", "+15% aura", 1, 4));
            w.setInfallible(null);
        }
        else if(nameOfSpell.equals("balance blade")) {
            System.out.println("Adding to list of positive charms.");
            w.getPositiveCharms().add("Balance Blade +25%");
            w.getBlades().put("universal", "Balance Blade +25&");
        }
        else if(nameOfSpell.equals("bladestorm")) {
            System.out.println("Adding bladestorm +20% to all allies");
            w.getPositiveCharms().add("Bladestorm +20%");
            w.getBlades().put("universal", "Bladestorm +20%");
        }
        else if(nameOfSpell.equals("donate power")) {
            System.out.println("Casting donate pip which gives 2 pips to self.");
            int pips = w.getPips();
            pips += 2; 
            if(pips >= 14) {
                w.setPips(14);
            } else {
                w.setPips(pips);
            }
        }
        else if(nameOfSpell.equals("scion of balance")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 955;
            if(wt2.getPips() == 11) {
                damageOfSpell = damageOfSpell * 2;
            }
            int spellDamage = (int) (damageOfSpell * (1 + (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);  
            System.out.println("Dealt " + spellDamage + " damage on opponent.");
        }
        else if(nameOfSpell.equals("duststorm jinn")) {
                int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
                int damageOfSpell = 285;
                if(w.getShieldsList().size() >= 2 && wt2.getTrapsList().size() >= 2) {
                    damageOfSpell = 485;
                    w.getShieldsList().remove(0);
                    w.getShieldsList().remove(1);
                    wt2.getTrapsList().remove(0);
                    wt2.getTrapsList().remove(1);
                }
                int spellDamage = (int) (damageOfSpell * (1 + (playerDamage/100.0) * 1.5));
                int[] spellDamageArr = {spellDamage, spellDamage, spellDamage};
                
                spellDamageArr[0] = checkForBuffs(spellDamageArr[0], "balance", nameOfSpell, "ice", w, wt2);
                System.out.println("Dealt " + spellDamageArr[0] + "ice damage on opponent.");
                spellDamageArr[1] = checkForBuffs(spellDamageArr[1], "balance", nameOfSpell, "fire", w, wt2);
                System.out.println("Dealt " + spellDamageArr[1] + "fire damage on opponent.");
                spellDamageArr[2] = checkForBuffs(spellDamageArr[2], "balance", nameOfSpell, "storm", w, wt2);
                System.out.println("Dealt " + spellDamageArr[2] + "storm damage on opponent.");
        }
        else if(nameOfSpell.equals("judgment")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 100 * (w.getPips());
            int spellDamage = (int) (damageOfSpell * (1 + (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, "balance", w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
        }
        else if(nameOfSpell.equals("mana burn")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 85 * (wt2.getPips()); 
            int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
        }
        else if(nameOfSpell.equals("mockenspiel")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int low = 800; 
            int high = 920; 
            int result = (int)(Math.random() * (high - low + 1)) + low;
            int spellDamage = (int) (result * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            wt2.getWeaknesses().put("universal", "-35% weakness");
            wt2.getWeaknesses().put("universal", "-35% weakness");
            wt2.getWeaknessesList().add("-35% weakness");
            wt2.getWeaknessesList().add("-35% weakness");
        }
        else if(nameOfSpell.equals("power nova")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 665;
            int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
        }
        else if(nameOfSpell.equals("ra")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int low = 420;
            int high = 500;
            int result = (int)(Math.random() * (high - low + 1)) + low;
            int spellDamage = (int) (result * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            wt2.getWeaknessesList().add("-40% weakness");
            wt2.getWeaknesses().put("universal", "-40% weakness");
        }
        else if(nameOfSpell.equals("jinn's fortune")) {
            int countTrapsonW1 = w.getTrapsList().size();
            int countShieldsonW2 = wt2.getShieldsList().size();
            if(countTrapsonW1 == 1) {
                wt2.getTrapsList().add("+30% universal trap");
                wt2.getTraps().put("universal", "+30% trap");
                wt2.getShieldsAndTraps().add("+30% universal trap");
            }
            else if(countTrapsonW1 >= 2) {
                wt2.getTrapsList().add("+30% universal trap");
                wt2.getTraps().put("universal", "+30% trap");
                wt2.getShieldsAndTraps().add("+30% universal trap");
                wt2.getTrapsList().add("+30% universal trap");
                wt2.getTraps().put("universal", "+30% trap");
                wt2.getShieldsAndTraps().add("+30% universal trap");
            }
            if(countShieldsonW2 == 1) {
                w.getShieldsList().add("-50% universal shield");
                w.getShields().put("universal", "-50% shield");
                w.getShieldsAndTraps().add("-50% universal shield");
            }
            else if(countShieldsonW2 >= 2) {
                w.getShieldsList().add("-50% universal shield");
                w.getShields().put("universal", "-50% shield");
                w.getShieldsAndTraps().add("-50% universal shield");
                w.getShieldsList().add("-50% universal shield");
                w.getShields().put("universal", "-50% shield");
                w.getShieldsAndTraps().add("-50% universal shield");
            }
        }
        else if(nameOfSpell.equals("oni's shadow")) {
            int countBladesonW2 = wt2.getPositiveCharms().size();
            int countWeaknessesonW1 = w.getWeaknessesList().size();
            if(countBladesonW2 == 1) {
                w.getPositiveCharms().add("+25% universal blade");
                w.getBlades().put("universal'", "+25% blade");
            }
            else if(countBladesonW2 >= 2) {
                w.getPositiveCharms().add("+25% universal blade");
                w.getBlades().put("universal", "+25% blade");
                w.getPositiveCharms().add("+25% universal blade");
                w.getBlades().put("universal", "+25% blade");
            }
            if(countWeaknessesonW1 == 1) {
                wt2.getWeaknessesList().add("-25% universal weakness");
                wt2.getWeaknesses().put("universal", "-25% weakness");
                wt2.getNegativeCharms().add("-25% universal weakness");
            }
            else if(countWeaknessesonW1 >= 2) {
                wt2.getWeaknessesList().add("-25% universal weakness");
                wt2.getWeaknesses().put("universal", "-25% weakness");
                wt2.getNegativeCharms().add("-25% universal weakness");
                wt2.getWeaknessesList().add("-25% universal weakness");
                wt2.getWeaknesses().put("universal", "-25% weakness");
                wt2.getNegativeCharms().add("-25% universal weakness");
            }
        }
        else if(nameOfSpell.equals("reshuffle")) {
            Deck mainDeckReshuffle = w.getMainDeck();
            Deck tcDeckReshuffle = w.getTcDeck();
            List<Spell> mainDeckAfterReshuffle = mainDeckReshuffle.getSpells();
            List<Spell> tcDeckAfterReshuffle = tcDeckReshuffle.getSpells();

            for(int i = 0; i < mainDeckAfterReshuffle.size(); i++) {
                Spell spell = mainDeckAfterReshuffle.get(i);
                spell.setName(mainDeckReshuffle.getSpellNames().get(i));
            }

            for(int i = 0; i < tcDeckAfterReshuffle.size(); i++) {
                Spell spell = tcDeckAfterReshuffle.get(i);
                spell.setName(tcDeckReshuffle.getSpellNames().get(i));
            }
        }
        else if(nameOfSpell.equals("sabertooth")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int low = 805;
            int high = 885;
            int result = (int)(Math.random() * (high - low + 1)) + low;
            int spellDamage = (int) (result * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
            w.getShieldsList().add("-50% life shield");
            w.getShieldsList().add("-50% death shield");
            w.getShieldsList().add("-50% myth shield");
            w.getShieldsList().add("-50% ice shield");
            w.getShieldsList().add("-50% fire shield");
            w.getShieldsList().add("-50% storm shield");
            w.getShields().put("life", "-50% life shield");
            w.getShields().put("death", "-50% death shield");
            w.getShields().put("myth", "-50% myth shield");
            w.getShields().put("ice", "-50% ice shield");
            w.getShields().put("fire", "-50% fire shield");
            w.getShields().put("storm", "-50% storm shield");
            w.getShieldsAndTraps().add("-50% life shield");
            w.getShieldsAndTraps().add("-50% death shield");
            w.getShieldsAndTraps().add("-50% myth shield");
            w.getShieldsAndTraps().add("-50% ice shield");
            w.getShieldsAndTraps().add("-50% fire shield");
            w.getShieldsAndTraps().add("-50% storm shield");
        }
        else if(nameOfSpell.equals("sandstorm")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int low = 255;
            int high = 295;
            int result = (int)(Math.random() * (high - low + 1)) + low;
            int spellDamage = (int) (result * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
        }
        else if(nameOfSpell.equals("spectral blast")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            String[] arr = {"ice", "fire", "storm"};
            Random rand = new Random();

            String randomElement = arr[rand.nextInt(arr.length)];
            System.out.println(randomElement);
            if(randomElement.equals("ice")) {
                int damageOfSpell = 365;
                int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
                spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, "ice", w, wt2);
                System.out.println("Dealt " + spellDamage + " on opponent.");
            } else if(randomElement.equals("fire")) {
                int damageOfSpell = 440;
                int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
                spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, "fire", w, wt2);
                System.out.println("Dealt " + spellDamage + " on opponent.");
            } else if(randomElement.equals("storm")) {
                int damageOfSpell = 550;
                int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
                spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, "storm", w, wt2);
            }
        }
        else if(nameOfSpell.equals("steel giant")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 55;
            int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            int[] arr = {120, 120, 120};
            wt2.getOvertimes().add(new Overtime("Steel Giant", arr, 2, 4, "balance", selected_.getSchool()));
        }
        else if(nameOfSpell.equals("supernova")) {
            if(wt2.getAura() != null) {
                int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
                int damageOfSpell = 405;
                int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
                spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
                System.out.println("Dealt " + spellDamage + " to opponent.");
                wt2.setAura(null);
            }
            else {
                System.out.println("no Aura. Dealt 0 damage.");
            }
        }
        else if(nameOfSpell.equals("elemental trap")) {
            wt2.getTrapsList().add("+25% ice trap");
            wt2.getTrapsList().add("+25% fire trap");
            wt2.getTrapsList().add("+25% storm trap");
            wt2.getShieldsAndTraps().add("+25% ice trap");
            wt2.getShieldsAndTraps().add("+25% fire trap");
            wt2.getShieldsAndTraps().add("+25% storm trap");
            wt2.getTraps().put("ice", "+25% ice trap");
            wt2.getTraps().put("fire", "+25% fire trap");
            wt2.getTraps().put("storm", "+25% storm trap");
        }
        else if(nameOfSpell.equals("shadow shrike")) {
            
        }
        else if(nameOfSpell.equals("wand-hit")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 180;
            int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
        }
        else if(nameOfSpell.equals("tc legion-shield")) {
            w.getShieldsList().add("tc legion-shield -45%");
            w.getShields().put("universal", "tc legion-shield -45%");
            w.getShieldsAndTraps().add("tc legion-shield -45%");
        }
        else if(nameOfSpell.equals("tc blinding light")) {
            System.out.println("Seeing whether opponent has stun effects.");
            if(wt2.getEffects().size() > 0) {
                wt2.getEffects().remove(0);
            } else {
                wt2.setIsStunned(true);
            }
        }
        else if(nameOfSpell.equals("tc infallible")) {
            w.setAura(null);
            w.setInfallible(new Infallible("infallible", 15, 1, 4));
        }
        else if(nameOfSpell.equals("tc empower")) {
            int pips = w.getPips();
            pips = pips -1;
            pips = pips + 3;
            w.setPips(pips);
        }
        else if(nameOfSpell.equals("cleanse charm")) {
            String removed = w.getNegativeCharms().remove(0); 
            if(removed.contains("weakness") && !w.getWeaknessesList().isEmpty()) {
                    w.getWeaknessesList().remove(0);
                    Map.Entry<String, String> first = w.getWeaknesses().entries().iterator().next();
                    w.getWeaknesses().remove(first.getKey(), first.getValue());
                }
                else if(removed.contains("infection") && !w.getInfections().isEmpty()) {
                    w.getInfections().remove(0);
                }
            System.out.println("Removed " + removed + " off self.");
        }
        else if(nameOfSpell.equals("tc balance blade")) {
            w.getPositiveCharms().add("tc balance blade");
            w.getBlades().put("blade", "tc balance blade");
        }
        else if(nameOfSpell.equals("tc power nova")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int damageOfSpell = 480;
            int spellDamage = (int) (damageOfSpell * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
            wt2.getWeaknessesList().add("-25% univeral weakness");
            wt2.getWeaknesses().put("universal", "-25% tc weakness");
            wt2.getNegativeCharms().add("-25% tc weakness");
        }
        else if(nameOfSpell.equals("tc reshuffle")) {
            Deck mainDeckReshuffle = w.getMainDeck();
            Deck tcDeckReshuffle = w.getTcDeck();
            List<Spell> mainDeckAfterReshuffle = mainDeckReshuffle.getSpells();
            List<Spell> tcDeckAfterReshuffle = tcDeckReshuffle.getSpells();

            for(int i = 0; i < mainDeckAfterReshuffle.size(); i++) {
                Spell spell = mainDeckAfterReshuffle.get(i);
                spell.setName(mainDeckReshuffle.getSpellNames().get(i));
            }

            for(int i = 0; i < tcDeckAfterReshuffle.size(); i++) {
                Spell spell = tcDeckAfterReshuffle.get(i);
                spell.setName(tcDeckReshuffle.getSpellNames().get(i));
            }
        }
        else if(nameOfSpell.equals("tc sabertooth")) {
            int playerDamage = w.getStatsInformation().getDamage().get(w.getIdentity().toLowerCase());
            int low = 1000;
            int high = 1100;
            int result = (int)(Math.random() * (high - low + 1)) + low;
            int spellDamage = (int) (result * (1+ (playerDamage/100.0) * 1.5));
            spellDamage = checkForBuffs(spellDamage, "balance", nameOfSpell, selected_.getSchool(), w, wt2);
            System.out.println("Dealt " + spellDamage + " on opponent.");
            w.getShieldsList().add("-50% tc life shield");
            w.getShieldsList().add("-50% tc death shield");
            w.getShieldsList().add("-50% tc myth shield");
            w.getShieldsList().add("-50% tc ice shield");
            w.getShieldsList().add("-50% tc fire shield");
            w.getShieldsList().add("-50% tc storm shield");
            w.getShields().put("life", "-50% tc life shield");
            w.getShields().put("death", "-50% tc death shield");
            w.getShields().put("myth", "-50% tc myth shield");
            w.getShields().put("ice", "-50% tc ice shield");
            w.getShields().put("fire", "-50% tc fire shield");
            w.getShields().put("storm", "-50% tc storm shield");
            w.getShieldsAndTraps().add("-50% tc life shield");
            w.getShieldsAndTraps().add("-50% tc death shield");
            w.getShieldsAndTraps().add("-50% tc myth shield");
            w.getShieldsAndTraps().add("-50% tc ice shield");
            w.getShieldsAndTraps().add("-50% tc fire shield");
            w.getShieldsAndTraps().add("-50% tc storm shield");
        }
        else if(nameOfSpell.equals("tc elemental shield")) {
            w.getShieldsList().add("-50% tc ice shield");
            w.getShieldsList().add("-50% tc fire shield");
            w.getShieldsList().add("tc -50% tc storm shield");
            w.getShields().put("ice", "-50% tc ice shield");
            w.getShields().put("fire", "-50% tc fire shield");
            w.getShields().put("storm", "-50% tc storm shield");
            w.getShieldsAndTraps().add("-50% tc ice shield");
            w.getShieldsAndTraps().add("-50% tc fire shield");
            w.getShieldsAndTraps().add("-50% tc storm shield");
        }
        else if(nameOfSpell.equals("tc elemental trap")) {
            wt2.getTrapsList().add("+25% tc ice trap");
            wt2.getTrapsList().add("+25% tc fire trap");
            wt2.getTrapsList().add("+25% tc storm trap");
            wt2.getTraps().put("ice", "+25% tc ice trap");
            wt2.getTraps().put("fire", "+25% tc fire trap");
            wt2.getTraps().put("storm", "+25% tc storm trap");
            wt2.getShieldsAndTraps().add("+25% tc ice trap");
            wt2.getShieldsAndTraps().add("+25% tc fire trap");
            wt2.getShieldsAndTraps().add("+25% tc storm trap");
        }
        else if(nameOfSpell.equals("tc weakness")) {
            wt2.getWeaknessesList().add("-25% tc weakness");
            wt2.getWeaknesses().put("universal", "-25% tc weakness");
            wt2.getNegativeCharms().add("-25% tc weakness");
        }
}   

    public void startRound() {
        if(t1Wizards.size() == 1) {
            System.out.println("Team 1's turn.");
            Wizard w = t1Wizards.get(0);
            Wizard wt2 = t2Wizards.get(0);
            
            if(w.getStatsInformation().getHealth() <= 0) {
                System.out.println(wt2.getName() + " wins the game.");
                logTurn("Team 2", roundNumber, "WON", 0.0);
                finalizeMatchLog("Team 2", roundNumber);
                winner = true;
                updateAgentsAfterMatch();
                return;
            }
            if(wt2.getStatsInformation().getHealth() <= 0) {
                System.out.println(w.getName() + " wins the game.");
                logTurn("Team 1", roundNumber, "WON", 0.0);
                finalizeMatchLog("Team 1", roundNumber);
                winner = true;
                updateAgentsAfterMatch();
                return;
            }

            int countTCAndMain1 = countTCAndMainDeckSpells(w);
            System.out.println("Size of tc and main deck of team 1: " + countTCAndMain1);
            int countTCAndMain2 = countTCAndMainDeckSpells(wt2);
            System.out.println("Size of tc and main deck of team 2: " + countTCAndMain2);

            if(countTCAndMain1 == 0) {
                int count = 0;
                for(Spell spell: activeHand) {
                    if(!spell.getName().equals("X")) {
                       count++;
                    }
                }
                if(count == 0) {
                    System.out.println(wt2.getName() + " wins the game.");
                    logTurn("Team 2", roundNumber, "WON", 0.0);
                    finalizeMatchLog("Team 2", roundNumber);
                    winner = true;
                    updateAgentsAfterMatch();
                    return;
                }
            }
            if(countTCAndMain2 == 0) {
                System.out.println(w.getName() + " wins the game.");
                logTurn("Team 1", roundNumber, "WON", 0.0);
                finalizeMatchLog("Team 1", roundNumber);
                winner = true;
                updateAgentsAfterMatch();
                return;
            }
            
            System.out.println(w.getIdentity() + " wizard " + w.getName());
            System.out.println("Pips: " + w.getPips() + " for team 1");
            System.out.println("Pips: " + wt2.getPips() + " for team 2");
            String pipsByCount = "";
            if(initial == 1) {
                pipsByCount = w.getPipsByCount(initial);
                initial = 0; 
            } else {
                pipsByCount = w.getPipsByCount();
            }

            System.out.println("Checking to see whether next pip is a regular pip or power pip based on power pip percentage.");
            System.out.println("Pips By Count: " + pipsByCount);
            System.out.println("Health: " + w.getStatsInformation().getHealth());
            System.out.println("Positive Charms: " + w.getPositiveCharms());
            System.out.println("Negative Charms: " + w.getNegativeCharms());
            System.out.println("Shields on self: " + w.getShieldsList());
            System.out.println("Traps on self: "  + w.getTraps());
            System.out.println("Effects on self: " + w.getEffects());
            System.out.println("Shadow Gauge: " + w.getShadowGauge());
            System.out.println("Health: " + wt2.getStatsInformation().getHealth());
            System.out.println("Positive Charms: " + wt2.getPositiveCharms());
            System.out.println("Negative Charms: " + wt2.getNegativeCharms());
            System.out.println("Shields on self: " + wt2.getShieldsList());
            System.out.println("Traps on self: "  + wt2.getTraps());
            System.out.println("Effects on self: " + wt2.getEffects());
            System.out.println("Shadow Gauge: " + wt2.getShadowGauge());
            if(w.getAura() != null) {
                System.out.println("Auras on self: " + w.getAura().getName() + ", Round: " + w.getAura().getTick() + "/" + w.getAura().getRounds());
            }
            if(w.getInfallible() != null) {
                System.out.println("Infallible on self: " + w.getInfallible().getName() + " , Round: " + w.getInfallible().getTick() + "/" + w.getInfallible().getRoundNo());
            }
            if(wt2.getAura() != null) {
                System.out.println("Auras on self: " + wt2.getAura().getName() + ", Round: " + wt2.getAura().getTick() + "/" + wt2.getAura().getRounds());
            }
            if(wt2.getOvertimes().size() > 0) {
                for(Overtime overtime: wt2.getOvertimes()) {
                    int spellDamage = overtime.getDamagePerRound()[overtime.getTick()-2];
                    System.out.println("Dealt " + spellDamage + " on tick " + overtime.getTick() + " out of round " + overtime.getRounds());
                }
            }
            
            Deck mainDeck = w.getMainDeck();
            Deck tcDeck = w.getTcDeck();

            List<Spell> mainDeckSpells = mainDeck.getSpells();
            List<Spell> tcDeckSpells = tcDeck.getSpells();

            Random r = new Random();

            int need = number - chosen.size();
            if (need > 0) {
                List<Integer> pool = IntStream.range(0, mainDeckSpells.size())
                    .filter(i -> {
                        String name = mainDeckSpells.get(i).getName();
                        return !"X".equals(name) && !chosen.contains(i);
                    })
                    .boxed()
                    .collect(Collectors.toList());

                if (pool.isEmpty()) {
                    System.out.println("Main deck is out — draw TC instead!");
                } else {
                    Collections.shuffle(pool, r);
                    int take = Math.min(need, pool.size());
                    for (int i = 0; i < take; i++) {
                        chosen.add(pool.get(i));
                    }
                    if (chosen.size() < number) {
                        System.out.println("Main deck ran out mid-draw — draw TC for the rest!");
                    }
                }
            }
            
            List<Integer> handIndices = new ArrayList<>(chosen);

            if(firstIteration) {
                System.out.println("SPELLS FOR: " + w.getName());
                for (int i = 0; i < handIndices.size(); i++) {
                    System.out.println("SPELL " + (i + 1) + ": " + mainDeckSpells.get(handIndices.get(i)).getName());
                    activeHand.add(mainDeckSpells.get(handIndices.get(i)));
                }
                firstIteration = false;
            } else {
                int ptr = 0;
                for (int i = 0; i < activeHand.size() && ptr < handIndices.size(); i++) {
                    Spell s = activeHand.get(i);
                    if ("X".equals(s.getName())) {
                        activeHand.set(i, mainDeckSpells.get(handIndices.get(ptr++)));
                    }
                }
                System.out.println("SPELLS FOR: " + w.getName());
                for (int i = 0; i < activeHand.size(); i++) {
                    System.out.println("SPELL " + (i + 1) + ": " + activeHand.get(i).getName());
                }
            }

            List<Integer> discardIndices = new ArrayList<>();
            
            if (usePPOForTeam1) {
                int discardCount = getPPOActionDiscard(w, wt2, activeHand);
                System.out.println("Team 1 AI: Discarding " + discardCount + " cards");
                
                discardIndices = selectCardsToDiscard(activeHand, w, discardCount);
                
                System.out.print("Team 1 AI discarding: ");
                for (int idx : discardIndices) {
                    System.out.print((idx + 1) + " ");
                }
                System.out.println();
                
                if (logMatches) {
                    logTurn("Team 1", roundNumber, "Discard " + discardCount + " cards", 0.0);
                }
            } else {
                System.out.println("Select which cards to discard (enter numbers separated by spaces):");
                
                String line = sc.nextLine();
                String[] parts = line.trim().split("\\s+");
                
                for (String part : parts) {
                    try {
                        int num = Integer.parseInt(part);
                        if (num >= 1 && num <= activeHand.size()) {
                            discardIndices.add(num - 1);
                        }
                    } catch (NumberFormatException e) {
                        System.out.println("Invalid input: " + part);
                    }
                }
            }

            Collections.sort(discardIndices);

            for (int i = 0; i < discardIndices.size(); i++) {
                int idx = discardIndices.get(i);
                if (idx < 0 || idx >= activeHand.size()) continue;

                int mdIdx = (idx < handIndices.size()) ? handIndices.get(idx) : -1;
                if (mdIdx >= 0) {
                    if(activeHand.get(idx).getName().contains("TC")) {
                        if (mdIdx < tcDeckSpells.size()) {
                            Spell tcSpell = tcDeckSpells.get(mdIdx); 
                            tcSpell.setName("X"); 
                            tcDeckSpells.set(mdIdx, tcSpell); 
                            chosen.remove(mdIdx);
                        }
                    } else {
                        if (mdIdx < mainDeckSpells.size()) {
                            Spell mainSpell = mainDeckSpells.get(mdIdx);
                            mainSpell.setName("X");
                            mainDeckSpells.set(mdIdx, mainSpell);
                            chosen.remove(mdIdx);
                        }
                    }   
                }

                Spell s = activeHand.get(idx);
                s.setName("X");
                activeHand.set(idx, s);
            }

            System.out.println("Remaining cards in hand (after discard):");
            for (int i = 0; i < activeHand.size(); i++) {
                System.out.println("SPELL " + (i + 1) + ": " + activeHand.get(i).getName());
            }

            int handX = (int) activeHand.stream().filter(s -> "X".equals(s.getName())).count();
            int tcAvailable = (int) tcDeckSpells.stream().filter(s -> !"X".equals(s.getName())).count();
            int maxDraw = Math.min(handX, tcAvailable);

            List<Integer> emptySlots = IntStream.range(0, activeHand.size())
                    .filter(i -> "X".equals(activeHand.get(i).getName()))
                    .boxed()
                    .collect(Collectors.toList());

            int drawCount = 0;

            if (usePPOForTeam1) {
                drawCount = getPPOActionDraw(w, wt2, activeHand, maxDraw);
                System.out.println("Team 1 AI: Drawing " + drawCount + " TCs (max: " + maxDraw + ")");
                
                if (logMatches) {
                    logTurn("Team 1", roundNumber, "Draw " + drawCount + " TCs", 0.0);
                }
            } else {
                boolean valid = false;
                while (!valid) {
                    System.out.println("Select how many cards to draw: 0-" + maxDraw);
                    String nLine = sc.nextLine().trim();
                    try {
                        drawCount = Integer.parseInt(nLine);
                        if (drawCount >= 0 && drawCount <= maxDraw) {
                            valid = true;
                        } else {
                            System.out.println("Invalid number. Please enter between 0 and " + maxDraw);
                        }
                    } catch (NumberFormatException e) {
                        System.out.println("Please enter a number.");
                    }
                }
            }

            Set<Integer> selectedTc = new HashSet<>();
            Random r1 = new Random();
            while (selectedTc.size() < drawCount) {
                int num = r1.nextInt(tcDeckSpells.size());
                if (!"X".equals(tcDeckSpells.get(num).getName())) {
                    selectedTc.add(num);
                }
            }

            List<Integer> drawnTc = new ArrayList<>(selectedTc);

            for (int i = 0; i < drawnTc.size(); i++) {
                int handIdx = emptySlots.get(i);
                Spell tc = tcDeckSpells.get(drawnTc.get(i));
                Spell copy = new Spell(tc.getName(), tc.getPips(), tc.getPipChance(), tc.getSchool(), tc.getDescriptino());
                activeHand.set(handIdx, copy);
                tc.setName("X");
            }

            System.out.println("Current hand after TC draw:");
            for (int i = 0; i < activeHand.size(); i++) {
                System.out.println("SPELL " + (i + 1) + ": " + activeHand.get(i).getName());
            }

            int castChoice = -1;
            int previousPlayerHealth = w.getStatsInformation().getHealth();
            int previousOpponentHealth = wt2.getStatsInformation().getHealth();

            if (usePPOForTeam1) {
                while(true) {
                    int ppoAction = getPPOAction(w, wt2, activeHand);
                    System.out.println("Team 1 AI selected action: " + ppoAction);
                    
                    if (ppoAction == 0) {
                        System.out.println("Team 1 AI: Passing.");
                        castChoice = 0;
                        if (logMatches) logTurn("Team 1", roundNumber, "Pass", 0.0);
                        break;
                    } else if (ppoAction >= 1 && ppoAction <= activeHand.size()) {
                        castChoice = ppoAction;
                        int slot = castChoice - 1;
                        Spell selected_ = activeHand.get(slot);
                        
                        if (selected_.getName().equals("X")) {
                            System.out.println("WARNING: Team 1 AI chose invalid X card. Passing instead.");
                            castChoice = 0;
                            break;
                        } else if (selected_.getPips() > w.getPips()) {
                            System.out.println("WARNING: Team 1 AI: Not enough pips. Passing instead.");
                            castChoice = 0;
                            break;
                        } else if ((selected_.getName().equals("Mockenspiel") || selected_.getName().equals("Shadow Shrike")) 
                                   && w.getShadowGauge() < 100) {
                            System.out.println("WARNING: Team 1 AI: Can't cast shadow spell. Passing instead.");
                            castChoice = 0;
                            break;
                        } else {
                            System.out.println("Team 1 AI: Casting " + selected_.getName());
                            if (logMatches) logTurn("Team 1", roundNumber, "Cast " + selected_.getName(), 0.0);
                            break;
                        }
                    } else {
                        System.out.println("WARNING: Team 1 AI returned out-of-range action. Passing.");
                        castChoice = 0;
                        break;
                    }
                }
            } else {
                while (true) {
                    System.out.println("Select a card to CAST (1-7), or 0 to skip:");
                    String castLine = sc.nextLine().trim();
                    try {
                        castChoice = Integer.parseInt(castLine);
                        if (castChoice > 0 && castChoice <= activeHand.size()) {
                            int slot = castChoice - 1; 
                            Spell selected_ = activeHand.get(slot);
                            if(selected_.getName().equals("X")) {
                                System.out.println("That spell is X'd out.");
                                continue;
                            }
                            if(selected_.getPips() <= w.getPips()) {
                                break;
                            } else {
                                System.out.println("You do not have enough pips to cast that spell.");
                            }
                        } else if(castChoice == 0) {
                            System.out.println("Passing.");
                            break;
                        }
                    } catch (NumberFormatException ignore) {}
                    System.out.println("Enter a number 0-" + activeHand.size() + ":");
                }
            }

            if (castChoice > 0) {
                int slot = castChoice - 1;
                Spell selected_ = activeHand.get(slot);
                if (selected_ == null || "X".equals(selected_.getName())) {
                    System.out.println("That slot is empty. No spell cast.");
                } else {
                    String nameOfSpell = selected_.getName().toLowerCase();
                    markCardUsedInDecks(selected_, mainDeckSpells, tcDeckSpells);
                    System.out.println("Name Of Spell: " + selected_.getName().toLowerCase());
                    selected_.setName("X");
                    activeHand.set(slot, selected_);

                    System.out.println("Casted: " + nameOfSpell + " (now marked X)");
                    int pipsAfterCast = w.getPips() - selected_.getPips(); 
                    w.setPips(pipsAfterCast);
                    operateSpellConditions(nameOfSpell, selected_, w, wt2);
                }
            }

            // Team 1 reward calculation
            if (usePPOForTeam1) {
                boolean gameOver = w.getStatsInformation().getHealth() <= 0 || 
                                   wt2.getStatsInformation().getHealth() <= 0;
                boolean spellWasCast = (castChoice > 0);
                double reward = calculateReward(w, wt2, previousPlayerHealth, previousOpponentHealth, spellWasCast);
                
                sendRewardToPPO(reward, gameOver);
                lastRewardTeam1 = reward;
                
                if (logMatches && (previousOpponentHealth != wt2.getStatsInformation().getHealth() || 
                                   previousPlayerHealth != w.getStatsInformation().getHealth())) {
                    int damageDealt = previousOpponentHealth - wt2.getStatsInformation().getHealth();
                    int damageTaken = previousPlayerHealth - w.getStatsInformation().getHealth();
                    System.out.println("Team 1 dealt: " + damageDealt + ", took: " + damageTaken);
                }
            }

            for (int i = 0; i < activeHand.size(); i++) {
                System.out.println("SPELL " + (i + 1) + ": " + activeHand.get(i).getName());
            }

            int activeHandSize = 0; 
            for(int i = 0; i < activeHand.size(); i++) {
                if(activeHand.get(i).getName().equals("X")) {
                    continue;
                } else {
                    activeHandSize++;
                }
            }
            number = 7 - activeHandSize;
            chosen.clear();

            if(w.getAura() != null) {
                w.getAura().setTick(w.getAura().getTick()+1);
                if(w.getAura().getTick() == w.getAura().getRounds()) {
                    w.setAura(null);
                }
            }
            offTurnPipGain(w);
            int add = (int)(Math.random() * (100 - w.getShadowGauge())) + 1;
            int newShadowGuage = w.getShadowGauge() + add;
            if(newShadowGuage > 200) {
                newShadowGuage = 200;
            }
            w.setShadowGauge(newShadowGuage);
        }

        if(t2Wizards.size() == 1) {
            System.out.println("Team 2's turn.");
            Wizard w = t1Wizards.get(0);
            Wizard wt2 = t2Wizards.get(0);
            
            if(w.getStatsInformation().getHealth() <= 0) {
                System.out.println(wt2.getName() + " wins the game.");
                logTurn("Team 2", roundNumber, "WON", 0.0);
                finalizeMatchLog("Team 2", roundNumber);
                winner = true;
                updateAgentsAfterMatch();
                return;
            }
            if(wt2.getStatsInformation().getHealth() <= 0) {
                System.out.println(w.getName() + " wins the game.");
                logTurn("Team 1", roundNumber, "WON", 0.0);
                finalizeMatchLog("Team 1", roundNumber);
                winner = true;
                updateAgentsAfterMatch();
                return;
            }

            int countTCAndMain1 = countTCAndMainDeckSpells(w);
            System.out.println("Size of tc and main deck of team 1: " + countTCAndMain1);
            int countTCAndMain2 = countTCAndMainDeckSpells(wt2);
            System.out.println("Size of tc and main deck of team 2: " + countTCAndMain2);

            if(countTCAndMain2 == 0) {
                int count = 0;
                for(Spell spell: activeHandTeam2) {
                    if(!spell.getName().equals("X")) {
                       count++;
                    }
                }
                if(count == 0) {
                    System.out.println(w.getName() + " wins the game.");
                    logTurn("Team 1", roundNumber, "WON", 0.0);
                    finalizeMatchLog("Team 1", roundNumber);
                    winner = true;
                    updateAgentsAfterMatch();
                    return;
                }
            }
            if(countTCAndMain1 == 0) {
                System.out.println(wt2.getName() + " wins the game.");
                logTurn("Team 2", roundNumber, "WON", 0.0);
                finalizeMatchLog("Team 2", roundNumber);
                winner = true;
                updateAgentsAfterMatch();
                return;
            }
            
            System.out.println(wt2.getIdentity() + " wizard " + wt2.getName());
            System.out.println("Pips: " + wt2.getPips() + " for team 2");
            System.out.println("Pips: " + w.getPips() + " for team 1");

            System.out.println("Checking to see whether next pip is a regular pip or power pip based on power pip percentage.");
            System.out.println("Health: " + wt2.getStatsInformation().getHealth());
            System.out.println("Positive Charms: " + wt2.getPositiveCharms());
            System.out.println("Negative Charms: " + wt2.getNegativeCharms());
            System.out.println("Shields on self: " + wt2.getShieldsList());
            System.out.println("Traps on self: "  + wt2.getTraps());
            System.out.println("Effects on self: " + wt2.getEffects());
            System.out.println("Shadow Gauge: " + wt2.getShadowGauge());
            System.out.println("Health: " + w.getStatsInformation().getHealth());
            System.out.println("Positive Charms: " + w.getPositiveCharms());
            System.out.println("Negative Charms: " + w.getNegativeCharms());
            System.out.println("Shields on self: " + w.getShieldsList());
            System.out.println("Traps on self: "  + w.getTraps());
            System.out.println("Effects on self: " + w.getEffects());
            System.out.println("Shadow Gauge: " + w.getShadowGauge());
            if(wt2.getAura() != null) {
                System.out.println("Auras on self: " + wt2.getAura().getName() + ", Round: " + wt2.getAura().getTick() + "/" + wt2.getAura().getRounds());
            }
            if(wt2.getInfallible() != null) {
                System.out.println("Infallible on self: " + wt2.getInfallible().getName() + " , Round: " + wt2.getInfallible().getTick() + "/" + wt2.getInfallible().getRoundNo());
            }
            if(w.getAura() != null) {
                System.out.println("Auras on self: " + w.getAura().getName() + ", Round: " + w.getAura().getTick() + "/" + w.getAura().getRounds());
            }
            if(w.getOvertimes().size() > 0) {
                for(Overtime overtime: w.getOvertimes()) {
                    int spellDamage = overtime.getDamagePerRound()[overtime.getTick()-2];
                    System.out.println("Dealt " + spellDamage + " on tick " + overtime.getTick() + " out of round " + overtime.getRounds());
                }
            }
            
            Deck mainDeck = wt2.getMainDeck();
            Deck tcDeck = wt2.getTcDeck();

            List<Spell> mainDeckSpells = mainDeck.getSpells();
            List<Spell> tcDeckSpells = tcDeck.getSpells();

            Random r = new Random();

            int need = numbert2 - chosenTeam2.size();
            if (need > 0) {
                List<Integer> pool = IntStream.range(0, mainDeckSpells.size())
                    .filter(i -> {
                        String name = mainDeckSpells.get(i).getName();
                        return !"X".equals(name) && !chosenTeam2.contains(i);
                    })
                    .boxed()
                    .collect(Collectors.toList());

                if (pool.isEmpty()) {
                    System.out.println("Main deck is out — draw TC instead!");
                } else {
                    Collections.shuffle(pool, r);
                    int take = Math.min(need, pool.size());
                    for (int i = 0; i < take; i++) {
                        chosenTeam2.add(pool.get(i));
                    }
                    if (chosenTeam2.size() < numbert2) {
                        System.out.println("Main deck ran out mid-draw — draw TC for the rest!");
                    }
                }
            }
            
            List<Integer> handIndices = new ArrayList<>(chosenTeam2);

            if(firstIterationTeam2) {
                System.out.println("SPELLS FOR: " + wt2.getName());
                for (int i = 0; i < handIndices.size(); i++) {
                    System.out.println("SPELL " + (i + 1) + ": " + mainDeckSpells.get(handIndices.get(i)).getName());
                    activeHandTeam2.add(mainDeckSpells.get(handIndices.get(i)));
                }
                firstIterationTeam2 = false;
            } else {
                int ptr = 0;
                for (int i = 0; i < activeHandTeam2.size() && ptr < handIndices.size(); i++) {
                    Spell s = activeHandTeam2.get(i);
                    if ("X".equals(s.getName())) {
                        activeHandTeam2.set(i, mainDeckSpells.get(handIndices.get(ptr++)));
                    }
                }
                System.out.println("SPELLS FOR: " + wt2.getName());
                for (int i = 0; i < activeHandTeam2.size(); i++) {
                    System.out.println("SPELL " + (i + 1) + ": " + activeHandTeam2.get(i).getName());
                }
            }

            List<Integer> discardIndices = new ArrayList<>();
            
            if (usePPOForTeam2) {
                int discardCount = getPPOActionDiscard(wt2, w, activeHandTeam2);
                System.out.println("Team 2 AI: Discarding " + discardCount + " cards");
                
                discardIndices = selectCardsToDiscard(activeHandTeam2, wt2, discardCount);
                
                System.out.print("Team 2 AI discarding: ");
                for (int idx : discardIndices) {
                    System.out.print((idx + 1) + " ");
                }
                System.out.println();
                
                if (logMatches) {
                    logTurn("Team 2", roundNumber, "Discard " + discardCount + " cards", 0.0);
                }
            } else {
                System.out.println("Select which cards to discard (enter numbers separated by spaces):");
                
                String line = sc.nextLine();
                String[] parts = line.trim().split("\\s+");
                
                for (String part : parts) {
                    try {
                        int num = Integer.parseInt(part);
                        if (num >= 1 && num <= activeHandTeam2.size()) {
                            discardIndices.add(num - 1);
                        }
                    } catch (NumberFormatException e) {
                        System.out.println("Invalid input: " + part);
                    }
                }
            }

            Collections.sort(discardIndices);

            for (int i = 0; i < discardIndices.size(); i++) {
                int idx = discardIndices.get(i);
                if (idx < 0 || idx >= activeHandTeam2.size()) continue;

                int mdIdx = (idx < handIndices.size()) ? handIndices.get(idx) : -1;
                if (mdIdx >= 0) {
                    Spell mainSpell = mainDeckSpells.get(mdIdx);
                    mainSpell.setName("X");
                    mainDeckSpells.set(mdIdx, mainSpell);
                    chosenTeam2.remove(mdIdx);
                }

                Spell s = activeHandTeam2.get(idx);
                s.setName("X");
                activeHandTeam2.set(idx, s);
            }

            System.out.println("Remaining cards in hand (after discard):");
            for (int i = 0; i < activeHandTeam2.size(); i++) {
                System.out.println("SPELL " + (i + 1) + ": " + activeHandTeam2.get(i).getName());
            }

            int handX = (int) activeHandTeam2.stream().filter(s -> "X".equals(s.getName())).count();
            int tcAvailable = (int) tcDeckSpells.stream().filter(s -> !"X".equals(s.getName())).count();
            int maxDraw = Math.min(handX, tcAvailable);

            List<Integer> emptySlots = IntStream.range(0, activeHandTeam2.size())
                    .filter(i -> "X".equals(activeHandTeam2.get(i).getName()))
                    .boxed()
                    .collect(Collectors.toList());

            int drawCount = 0;

            if (usePPOForTeam2) {
                drawCount = getPPOActionDraw(wt2, w, activeHandTeam2, maxDraw);
                System.out.println("Team 2 AI: Drawing " + drawCount + " TCs (max: " + maxDraw + ")");
                
                if (logMatches) {
                    logTurn("Team 2", roundNumber, "Draw " + drawCount + " TCs", 0.0);
                }
            } else {
                boolean valid = false;
                while (!valid) {
                    System.out.println("Select how many cards to draw: 0-" + maxDraw);
                    String nLine = sc.nextLine().trim();
                    try {
                        drawCount = Integer.parseInt(nLine);
                        if (drawCount >= 0 && drawCount <= maxDraw) {
                            valid = true;
                        } else {
                            System.out.println("Invalid number. Please enter between 0 and " + maxDraw);
                        }
                    } catch (NumberFormatException e) {
                        System.out.println("Please enter a number.");
                    }
                }
            }

            Set<Integer> selectedTc = new HashSet<>();
            Random r1 = new Random();
            while (selectedTc.size() < drawCount) {
                int num = r1.nextInt(tcDeckSpells.size());
                if (!"X".equals(tcDeckSpells.get(num).getName())) {
                    selectedTc.add(num);
                }
            }

            List<Integer> drawnTc = new ArrayList<>(selectedTc);

            for (int i = 0; i < drawnTc.size(); i++) {
                int handIdx = emptySlots.get(i);
                Spell tc = tcDeckSpells.get(drawnTc.get(i));
                Spell copy = new Spell(tc.getName(), tc.getPips(), tc.getPipChance(), tc.getSchool(), tc.getDescriptino());
                activeHandTeam2.set(handIdx, copy);
                tc.setName("X");
            }

            System.out.println("Current hand after TC draw:");
            for (int i = 0; i < activeHandTeam2.size(); i++) {
                System.out.println("SPELL " + (i + 1) + ": " + activeHandTeam2.get(i).getName());
            }

            int castChoice = -1;
            int previousPlayerHealth = wt2.getStatsInformation().getHealth();
            int previousOpponentHealth = w.getStatsInformation().getHealth();

            if (usePPOForTeam2) {
                while(true) {
                    int ppoAction = getPPOAction(wt2, w, activeHandTeam2);
                    System.out.println("Team 2 AI selected action: " + ppoAction);
                    
                    if (ppoAction == 0) {
                        System.out.println("Team 2 AI: Passing.");
                        castChoice = 0;
                        if (logMatches) logTurn("Team 2", roundNumber, "Pass", 0.0);
                        break;
                    } else if (ppoAction >= 1 && ppoAction <= activeHandTeam2.size()) {
                        castChoice = ppoAction;
                        int slot = castChoice - 1;
                        Spell selected_ = activeHandTeam2.get(slot);
                        
                        if (selected_.getName().equals("X")) {
                            System.out.println("WARNING: Team 2 AI chose invalid X card. Passing instead.");
                            castChoice = 0;
                            break;
                        } else if (selected_.getPips() > wt2.getPips()) {
                            System.out.println("WARNING: Team 2 AI: Not enough pips. Passing instead.");
                            castChoice = 0;
                            break;
                        } else if ((selected_.getName().equals("Mockenspiel") || selected_.getName().equals("Shadow Shrike")) 
                                   && wt2.getShadowGauge() < 100) {
                            System.out.println("WARNING: Team 2 AI: Can't cast shadow spell. Passing instead.");
                            castChoice = 0;
                            break;
                        } else {
                            System.out.println("Team 2 AI: Casting " + selected_.getName());
                            if (logMatches) logTurn("Team 2", roundNumber, "Cast " + selected_.getName(), 0.0);
                            break;
                        }
                    } else {
                        System.out.println("WARNING: Team 2 AI returned out-of-range action. Passing.");
                        castChoice = 0;
                        break;
                    }
                }
            } else {
                while (true) {
                    System.out.println("Select a card to CAST (1-7), or 0 to skip:");
                    String castLine = sc.nextLine().trim();
                    try {
                        castChoice = Integer.parseInt(castLine);
                        if (castChoice > 0 && castChoice <= activeHandTeam2.size()) {
                            int slot = castChoice - 1; 
                            Spell selected_ = activeHandTeam2.get(slot);
                            if(selected_.getName().equals("X")) {
                                System.out.println("That spell is X'd out.");
                                continue;
                            }
                            if(selected_.getPips() <= wt2.getPips()) {
                                break;
                            } else {
                                System.out.println("You do not have enough pips to cast that spell.");
                            }
                        } else if(castChoice == 0) {
                            System.out.println("Passing.");
                            break;
                        }
                    } catch (NumberFormatException ignore) {}
                    System.out.println("Enter a number 0-" + activeHandTeam2.size() + ":");
                }
            }

            if (castChoice > 0) {
                int slot = castChoice - 1;
                Spell selected_ = activeHandTeam2.get(slot);
                if (selected_ == null || "X".equals(selected_.getName())) {
                    System.out.println("That slot is empty. No spell cast.");
                } else {
                    String nameOfSpell = selected_.getName().toLowerCase();
                    markCardUsedInDecks(selected_, mainDeckSpells, tcDeckSpells);
                    System.out.println("Name Of Spell: " + selected_.getName().toLowerCase());
                    selected_.setName("X");
                    activeHandTeam2.set(slot, selected_);

                    System.out.println("Casted: " + nameOfSpell + " (now marked X)");
                    int pipsAfterCast = wt2.getPips() - selected_.getPips(); 
                    wt2.setPips(pipsAfterCast);
                    operateSpellConditions(nameOfSpell, selected_, wt2, w);
                }
            }

            // Team 2 reward calculation
            if (usePPOForTeam2) {
                boolean gameOver = w.getStatsInformation().getHealth() <= 0 || 
                                   wt2.getStatsInformation().getHealth() <= 0;
                boolean spellWasCast = (castChoice > 0);
                double reward = calculateReward(wt2, w, previousPlayerHealth, previousOpponentHealth, spellWasCast);
                
                sendRewardToPPO(reward, gameOver);
                lastRewardTeam2 = reward;
                
                if (logMatches && (previousOpponentHealth != w.getStatsInformation().getHealth() || 
                                   previousPlayerHealth != wt2.getStatsInformation().getHealth())) {
                    int damageDealt = previousOpponentHealth - w.getStatsInformation().getHealth();
                    int damageTaken = previousPlayerHealth - wt2.getStatsInformation().getHealth();
                    System.out.println("Team 2 dealt: " + damageDealt + ", took: " + damageTaken);
                }
            }

            for (int i = 0; i < activeHandTeam2.size(); i++) {
                System.out.println("SPELL " + (i + 1) + ": " + activeHandTeam2.get(i).getName());
            }

            int activeHandSize = 0; 
            for(int i = 0; i < activeHandTeam2.size(); i++) {
                if(activeHandTeam2.get(i).getName().equals("X")) {
                    continue;
                } else {
                    activeHandSize++;
                }
            }
            numbert2 = 7 - activeHandSize;
            chosenTeam2.clear();

            if(wt2.getAura() != null) {
                wt2.getAura().setTick(wt2.getAura().getTick()+1);
                if(wt2.getAura().getTick() == wt2.getAura().getRounds()) {
                    wt2.setAura(null);
                }
            }
            offTurnPipGain(wt2);
            int add = (int)(Math.random() * (100 - wt2.getShadowGauge())) + 1;
            int newShadowGuage = wt2.getShadowGauge() + add;
            if(newShadowGuage > 200) {
                newShadowGuage = 200;
            }
            wt2.setShadowGauge(newShadowGuage);
        }
    }

    // ========== PPO AGENT INTEGRATION METHODS ==========
    
    private String createGameState(Wizard player, Wizard opponent, List<Spell> hand) {
        StringBuilder json = new StringBuilder();
        json.append("{");
        
        json.append("\"player_health\":").append(player.getStatsInformation().getHealth()).append(",");
        json.append("\"player_pips\":").append(player.getPips()).append(",");
        json.append("\"player_shadow_gauge\":").append(player.getShadowGauge()).append(",");
        
        json.append("\"opponent_health\":").append(opponent.getStatsInformation().getHealth()).append(",");
        json.append("\"opponent_pips\":").append(opponent.getPips()).append(",");
        json.append("\"opponent_shadow_gauge\":").append(opponent.getShadowGauge()).append(",");
        
        json.append("\"hand\":[");
        for (int i = 0; i < hand.size(); i++) {
            if (i > 0) json.append(",");
            Spell spell = hand.get(i);
            json.append("{");
            
            boolean isValid = !spell.getName().equals("X");
            json.append("\"is_valid\":").append(isValid).append(",");
            json.append("\"pips\":").append(spell.getPips()).append(",");
            
            int damagePotential = 0;
            if (isValid && spell.getPips() <= player.getPips()) {
                damagePotential = spell.getPips() * 100;
            }
            json.append("\"damage_potential\":").append(damagePotential);
            json.append("}");
        }
        json.append("],");
        
        json.append("\"has_aura\":").append(player.getAura() != null).append(",");
        json.append("\"opponent_has_aura\":").append(opponent.getAura() != null).append(",");
        
        json.append("\"round_number\":").append(roundNumber).append(",");
        
        int cardsRemaining = 0;
        for (Spell s : player.getMainDeck().getSpells()) {
            if (!s.getName().equals("X")) cardsRemaining++;
        }
        for (Spell s : player.getTcDeck().getSpells()) {
            if (!s.getName().equals("X")) cardsRemaining++;
        }
        json.append("\"cards_remaining\":").append(cardsRemaining).append(",");
        
        json.append("\"valid_actions\":[1");
        for (int i = 0; i < hand.size(); i++) {
            Spell spell = hand.get(i);
            boolean canCast = !spell.getName().equals("X") && spell.getPips() <= player.getPips();
            
            if (spell.getName().equals("Mockenspiel") || spell.getName().equals("Shadow Shrike")) {
                canCast = canCast && player.getShadowGauge() >= 100;
            }
            
            json.append(",").append(canCast ? 1 : 0);
        }
        json.append("],");
        json.append("\"training\":").append(trainingMode);
        json.append("}");
        
        return json.toString();
    }
    
    private int callPPOAgent(String command, String input) {
        try {
            ProcessBuilder pb = new ProcessBuilder(pythonPath, ppoScriptPath, command);
            pb.redirectErrorStream(true);
            
            Process process = pb.start();
            
            if (input != null && !input.isEmpty()) {
                try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
                    writer.write(input);
                    writer.flush();
                }
            }
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line);
            }
            
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                String response = output.toString();
                int actionIndex = response.indexOf("\"action\":");
                if (actionIndex != -1) {
                    String actionPart = response.substring(actionIndex + 9);
                    actionPart = actionPart.replaceAll("[^0-9]", "");
                    if (!actionPart.isEmpty()) {
                        return Integer.parseInt(actionPart);
                    }
                }
            } else {
                System.err.println("PPO Agent failed with exit code: " + exitCode);
                System.err.println("Output: " + output.toString());
            }
        } catch (Exception e) {
            System.err.println("Error calling PPO agent: " + e.getMessage());
            e.printStackTrace();
        }
        
        return 0;
    }
    
    private int getPPOAction(Wizard player, Wizard opponent, List<Spell> hand) {
        String gameState = createGameState(player, opponent, hand);
        return callPPOAgent("select_action", gameState);
    }
    
    private int getPPOActionDiscard(Wizard player, Wizard opponent, List<Spell> hand) {
        String gameState = createGameStateForDiscard(player, opponent, hand);
        int action = callPPOAgent("select_action", gameState);
        return Math.max(0, Math.min(hand.size(), action));
    }
    
    private int getPPOActionDraw(Wizard player, Wizard opponent, List<Spell> hand, int maxDraw) {
        String gameState = createGameStateForDraw(player, opponent, hand, maxDraw);
        int action = callPPOAgent("select_action", gameState);
        return Math.max(0, Math.min(maxDraw, action));
    }
    
    private String createGameStateForDiscard(Wizard player, Wizard opponent, List<Spell> hand) {
        StringBuilder json = new StringBuilder();
        json.append("{");
        
        json.append("\"player_health\":").append(player.getStatsInformation().getHealth()).append(",");
        json.append("\"player_pips\":").append(player.getPips()).append(",");
        json.append("\"opponent_health\":").append(opponent.getStatsInformation().getHealth()).append(",");
        json.append("\"opponent_pips\":").append(opponent.getPips()).append(",");
        
        int castableCards = 0;
        int uncastableCards = 0;
        int totalPipCost = 0;
        
        for (Spell spell : hand) {
            if (!spell.getName().equals("X")) {
                totalPipCost += spell.getPips();
                if (spell.getPips() <= player.getPips()) {
                    castableCards++;
                } else {
                    uncastableCards++;
                }
            }
        }
        
        json.append("\"castable_cards\":").append(castableCards).append(",");
        json.append("\"uncastable_cards\":").append(uncastableCards).append(",");
        json.append("\"avg_pip_cost\":").append(hand.size() > 0 ? totalPipCost / hand.size() : 0).append(",");
        json.append("\"hand_size\":").append(hand.size()).append(",");
        json.append("\"round_number\":").append(roundNumber).append(",");
        
        json.append("\"decision_type\":\"discard\",");
        
        json.append("\"valid_actions\":[");
        for (int i = 0; i <= hand.size(); i++) {
            if (i > 0) json.append(",");
            json.append("1");
        }
        json.append("],");
        
        json.append("\"training\":").append(trainingMode);
        json.append("}");
        
        return json.toString();
    }
    
    private String createGameStateForDraw(Wizard player, Wizard opponent, List<Spell> hand, int maxDraw) {
        StringBuilder json = new StringBuilder();
        json.append("{");
        
        json.append("\"player_health\":").append(player.getStatsInformation().getHealth()).append(",");
        json.append("\"player_pips\":").append(player.getPips()).append(",");
        json.append("\"opponent_health\":").append(opponent.getStatsInformation().getHealth()).append(",");
        json.append("\"opponent_pips\":").append(opponent.getPips()).append(",");
        
        int emptySlots = 0;
        int filledSlots = 0;
        for (Spell spell : hand) {
            if (spell.getName().equals("X")) {
                emptySlots++;
            } else {
                filledSlots++;
            }
        }
        
        json.append("\"empty_slots\":").append(emptySlots).append(",");
        json.append("\"filled_slots\":").append(filledSlots).append(",");
        json.append("\"max_draw\":").append(maxDraw).append(",");
        json.append("\"round_number\":").append(roundNumber).append(",");
        
        json.append("\"decision_type\":\"draw\",");
        
        json.append("\"valid_actions\":[");
        for (int i = 0; i <= maxDraw && i <= 7; i++) {
            if (i > 0) json.append(",");
            json.append("1");
        }
        for (int i = maxDraw + 1; i <= 7; i++) {
            json.append(",0");
        }
        json.append("],");
        
        json.append("\"training\":").append(trainingMode);
        json.append("}");
        
        return json.toString();
    }
    
    private List<Integer> selectCardsToDiscard(List<Spell> hand, Wizard player, int count) {
        List<Integer> toDiscard = new ArrayList<>();
        
        // Get indices of non-empty cards
        List<Integer> availableIndices = new ArrayList<>();
        for (int i = 0; i < hand.size(); i++) {
            if (!hand.get(i).getName().equals("X")) {
                availableIndices.add(i);
            }
        }
        
        // Randomly select 'count' cards to discard
        Collections.shuffle(availableIndices);
        for (int i = 0; i < Math.min(count, availableIndices.size()); i++) {
            toDiscard.add(availableIndices.get(i));
        }
        
        return toDiscard;
    }
    
    private static class CardScore {
        int index;
        double score;
        
        CardScore(int index, double score) {
            this.index = index;
            this.score = score;
        }
    }
    
    private void sendRewardToPPO(double reward, boolean done) {
        try {
            String rewardData = "{\"reward\":" + reward + ",\"done\":" + done + "}";
            
            ProcessBuilder pb = new ProcessBuilder(pythonPath, ppoScriptPath, "store_reward");
            pb.redirectErrorStream(true);
            
            Process process = pb.start();
            
            try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()))) {
                writer.write(rewardData);
                writer.flush();
            }
            
            process.waitFor();
        } catch (Exception e) {
            System.err.println("Error sending reward: " + e.getMessage());
        }
    }
    
    private void updatePPOAgent(String team) {
        try {
            ProcessBuilder pb = new ProcessBuilder(pythonPath, ppoScriptPath, "update", team);
            pb.redirectErrorStream(true);
            
            Process process = pb.start();
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            boolean hadOutput = false;
            
            while ((line = reader.readLine()) != null) {
                System.out.println("  [" + team + "] " + line);
                hadOutput = true;
            }
            
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                if (!hadOutput) {
                    System.out.println("  ⚠ " + team + " update completed but no output received");
                }
                System.out.println("  ✓ " + team + " agent updated successfully\n");
            } else {
                System.err.println("  ✗ " + team + " agent update failed with exit code: " + exitCode);
                System.err.println("  Check ppo_agent.py for errors\n");
            }
            
        } catch (IOException e) {
            System.err.println("  ✗ Failed to start Python process for " + team);
            System.err.println("  Make sure ppo_agent.py exists and Python is installed");
            System.err.println("  Error: " + e.getMessage() + "\n");
        } catch (InterruptedException e) {
            System.err.println("  ✗ Update process interrupted for " + team);
            Thread.currentThread().interrupt();
        } catch (Exception e) {
            System.err.println("  ✗ Unexpected error updating " + team + ": " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void savePPOModel(String path) {
        try {
            new File("checkpoints").mkdirs();
            
            ProcessBuilder pb = new ProcessBuilder(pythonPath, ppoScriptPath, "save_model", path);
            pb.redirectErrorStream(true);
            
            Process process = pb.start();
            int exitCode = process.waitFor();
            
            if (exitCode == 0) {
                System.out.println("  ✓ Model saved: " + path);
            } else {
                System.err.println("  ✗ Failed to save model: " + path);
            }
        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
        }
    }
    
    private double calculateReward(Wizard player, Wizard opponent, 
                                    int previousPlayerHealth, int previousOpponentHealth, 
                                    boolean spellCast) {
        double reward = 0.0;
        
        int opponentDamage = previousOpponentHealth - opponent.getStatsInformation().getHealth();
        if (opponentDamage > 0) {
            reward += opponentDamage / 100.0;
        }
        
        int playerDamage = previousPlayerHealth - player.getStatsInformation().getHealth();
        if (playerDamage > 0) {
            reward -= playerDamage / 100.0;
        }
        
        if (spellCast) {
            reward += 0.1;
        }
        
        if (opponent.getStatsInformation().getHealth() <= 0) {
            reward += 10.0;
        }
        
        if (player.getStatsInformation().getHealth() <= 0) {
            reward -= 10.0;
        }
        
        return reward;
    }
    
    private double calculateDiscardReward(int discardedCount, List<Spell> handBefore, Wizard player) {
        return 0.0;
    }
    
    private double calculateDrawReward(int drewCount, int maxDraw, int emptySlots, int filledSlots) {
        return 0.0;
    }

    private int countTCAndMain(Deck mainDeck, Deck tcDeck) {
        List<Spell> mainDeckSpells = mainDeck.getSpells();
        List<Spell> tcDeckSpells = tcDeck.getSpells();

        int countTCAndMain = 0; 

        for (Spell spell : mainDeckSpells) {
            if (!spell.getName().equals("X")) {
                countTCAndMain++;
            }
        }

        for (Spell spell : tcDeckSpells) {
            if (!spell.getName().equals("X")) {
                countTCAndMain++;
            }
        }

        return countTCAndMain;
    }
    
    private void initializeMatchLog(Wizard w, Wizard wt2) {
        matchLog.setLength(0);
        matchLog.append("=".repeat(80)).append("\n");
        matchLog.append("MATCH #").append(matchNumber).append(" - ").append(java.time.LocalDateTime.now()).append("\n");
        matchLog.append("=".repeat(80)).append("\n");
        matchLog.append("Team 1 (").append(usePPOForTeam1 ? "AI" : "Human").append("): ");
        matchLog.append(w.getName()).append(" (").append(w.getIdentity()).append(")\n");
        matchLog.append("Team 2 (").append(usePPOForTeam2 ? "AI" : "Human").append("): ");
        matchLog.append(wt2.getName()).append(" (").append(wt2.getIdentity()).append(")\n");
        matchLog.append("-".repeat(80)).append("\n");
    }
    
    private void logTurn(String team, int round, String action, double reward) {
        matchLog.append(String.format("Round %d | %s | %s | Reward: %.2f\n", 
                                     round, team, action, reward));
    }
    
    private void finalizeMatchLog(String winner, int finalRound) {
        matchLog.append("-".repeat(80)).append("\n");
        matchLog.append(String.format("WINNER: %s | Final Round: %d\n", winner, finalRound));
        matchLog.append("=".repeat(80)).append("\n\n");
        
        if (logMatches) {
            try (java.io.FileWriter fw = new java.io.FileWriter(logFilePath, true)) {
                fw.write(matchLog.toString());
                
                if (winner.equals("Team 2")) {
                    try (java.io.FileWriter winFw = new java.io.FileWriter("wins_log_t2.txt", true)) {
                        winFw.write(matchLog.toString());
                    }
                }
                if (winner.equals("Team 1")) {
                    try (java.io.FileWriter winFw = new java.io.FileWriter("wins_log_t1.txt", true)) {
                        winFw.write(matchLog.toString());
                    }
                }
            } catch (java.io.IOException e) {
                System.err.println("Error writing match log: " + e.getMessage());
            }
        }
    }

    private void updateAgentsAfterMatch() {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("MATCH COMPLETE - Updating agents...");
        System.out.println("=".repeat(50));
        
        if (usePPOForTeam1 && trainingMode) {
            System.out.println("Updating Team 1 AI...");
            updatePPOAgent("team1");
        }
        
        if (usePPOForTeam2 && trainingMode) {
            System.out.println("Updating Team 2 AI...");
            updatePPOAgent("team2");
        }
        
            System.out.println("\n📦 Saving checkpoint backups...");
            
            if (usePPOForTeam1) {
                savePPOModel("checkpoints/team1_match_" + matchNumber + ".pth");
            }
            
            if (usePPOForTeam2) {
                savePPOModel("checkpoints/team2_match_" + matchNumber + ".pth");
            }
        
        System.out.println("=".repeat(50) + "\n");
    }

    public void printWizardTeamInformation() {
        for(Wizard wizard: t1Wizards) {
            System.out.println("Name: " + wizard.getName());
            System.out.println("Identity: " + wizard.getIdentity());
            System.out.println("Stats Information: " + wizard.getStatsInformation().toString());
            System.out.println("Main Deck Size: " + wizard.getMainDeck().getSpells().size());
            System.out.println("TC Deck Size: " + wizard.getTcDeck().getSpells().size());
        }
        for(Wizard wizard: t2Wizards) {
            System.out.println("Name: " + wizard.getName());
            System.out.println("Identity: " + wizard.getIdentity());
            System.out.println("Stats Information: " + wizard.getStatsInformation().toString());
            System.out.println("Main Deck Size: " + wizard.getMainDeck().getSpells().size());
            System.out.println("TC Deck Size: " + wizard.getTcDeck().getSpells().size());
        }
    }
}
