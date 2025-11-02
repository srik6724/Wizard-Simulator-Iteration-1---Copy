import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

public class Match_ {
    // ---------- Config ----------
    private static final int HAND_SIZE = 7;

    // ---------- IO / RNG ----------
    private final Scanner sc = new Scanner(System.in);
    private final Random rng = new Random();

    // ---------- Teams ----------
    private final List<Wizard> t1Wizards = new ArrayList<>();
    private final List<Wizard> t2Wizards = new ArrayList<>();

    // Track per-wizard hand state (persists across rounds)
    private final Map<Wizard, HandState> handMap = new HashMap<>();

    // Used once so first pip print uses initial distribution text
    private int initial = 1;

    // ---------- Hand state ----------
    private static final class HandState {
        // Fixed-size list of spells; null means empty slot to be filled next turn
        final List<Spell> slots = new ArrayList<>(HAND_SIZE);
        // How many cards to auto-draw at the start of next turn
        int pendingDraw = 0;

        HandState() {
            for (int i = 0; i < HAND_SIZE; i++) slots.add(null);
        }

        boolean isInitialized() {
            for (Spell s : slots) if (s != null) return true;
            return false;
        }

        void setSlot(int idx, Spell s) { slots.set(idx, s); }

        List<Integer> emptySlots() {
            List<Integer> res = new ArrayList<>();
            for (int i = 0; i < slots.size(); i++) if (slots.get(i) == null) res.add(i);
            return res;
        }
    }

    // ---------- Constructor ----------
    public Match_() {
        // Build teams from files (unchanged logic, but cleaned scanners)
        for (int i = 0; i < 2; i++) {
            System.out.println("Enter size of your team.");
            int size = safeNextInt();
            for (int y = 0; y < size; y++) {
                System.out.println("Enter name of your team member.");
                String name = sc.nextLine().trim();

                System.out.println("Enter identity for " + name + ":");
                String identity = sc.nextLine().trim();

                // Read stats
                StatsInformation info = readStats("stats_t" + (i + 1) + "_" + (y + 1) + ".txt");

                // Read decks
                List<Spell> mainDeckSpells = readDeck("deck_main_t" + (i + 1) + "_" + (y + 1) + ".txt", true);
                List<Spell> tcSpells       = readDeck("deck_tc_t"   + (i + 1) + "_" + (y + 1) + ".txt",   false);

                Deck mainDeck = new Deck(mainDeckSpells);
                Deck tcDeck   = new Deck(tcSpells);

                int pips = (i == 0) ? 3 : 5;

                Wizard w = new Wizard(name, identity, info, mainDeck, tcDeck, pips);
                if (i == 0) t1Wizards.add(w); else t2Wizards.add(w);

                // Prepare persistent hand state
                handMap.put(w, new HandState());
            }
        }

        printWizardTeamInformation();

        // Initialize hands once (first time only)
        for (Wizard w : t1Wizards) initHandIfNeeded(w);
        for (Wizard w : t2Wizards) initHandIfNeeded(w);

        // Run a few rounds as demo
        for (int r = 1; r <= 10; r++) {
            System.out.println("\n================= ROUND " + r + " =================");
            startRoundTeam(t1Wizards, "Team 1");
            startRoundTeam(t2Wizards, "Team 2");
        }
    }

    // ---------- Round flow for a team ----------
    private void startRoundTeam(List<Wizard> team, String label) {
        System.out.println(label + "'s turn.");
        for (Wizard w : team) {
            // 1) Auto-fill any pending draws from previous discards
            autoFillHandStartOfTurn(w);

            // 2) Show state
            System.out.println(w.getIdentity() + " wizard " + w.getName());
            System.out.println("Pips: " + w.getPips());
            String pipsByCount;
            if (initial == 1) {
                pipsByCount = w.getPipsByCount(1);
                initial = 0;
            } else {
                pipsByCount = w.getPipsByCount();
            }
            System.out.println("Pips By Count: " + pipsByCount);
            System.out.println("Health: " + w.getStatsInformation().getHealth());

            // 3) Show hand
            printHand(w);

            // 4) Discard flow (user may choose none)
            // 4) Discard flow (user may choose none)
List<Integer> discardSlots = askForDiscards();
applyDiscards(w, discardSlots);

// 5) Draw policy:
//    - Default: auto-draw next turn (pendingDraw)
//    - Optional: player can draw some/all now into the exact emptied slots
HandState hs = handMap.get(w);
hs.pendingDraw += discardSlots.size();                 // default: draw next turn
List<Integer> empties = hs.emptySlots();               // exact slots to fill

int drawNowMax = Math.min(empties.size(), discardSlots.size());
int drawNow = askImmediateDrawCount(drawNowMax);       // ENTER => 0

if (drawNow > 0) {
    // Fill those exact emptied slots now, left-to-right
    fillSpecificSlots(w, empties, drawNow);
    hs.pendingDraw -= drawNow;                         // remaining draws happen next turn
    System.out.println("Drew " + drawNow + " card(s) now. The rest will be drawn at the start of your next turn.");
}

// Show current hand state after optional draw
printHand(w);


            // 6) Pip roll for next turn’s pips (as in your code)
            rollAndApplyPip(w);
        }
    }

    // ---------- Initialize hand once ----------
    private void initHandIfNeeded(Wizard w) {
        HandState hs = handMap.get(w);
        if (hs.isInitialized()) return;

        // Draw 7 unique from MAIN deck to start
        List<Spell> main = w.getMainDeck().getSpells();
        if (main.isEmpty()) {
            System.out.println("Warning: " + w.getName() + " has empty main deck.");
            return;
        }

        Set<Integer> chosen = new HashSet<>();
        while (chosen.size() < HAND_SIZE && chosen.size() < main.size()) {
            chosen.add(rng.nextInt(main.size()));
        }

        int slot = 0;
        for (int idx : chosen) hs.setSlot(slot++, main.get(idx));
        // If main deck has fewer than HAND_SIZE, remaining slots stay null and will fill next turn
        hs.pendingDraw += Math.max(0, HAND_SIZE - chosen.size());
    }

    // ---------- Auto-fill at start of a turn ----------
    private void autoFillHandStartOfTurn(Wizard w) {
        HandState hs = handMap.get(w);
        if (hs.pendingDraw <= 0) return;

        List<Integer> empties = hs.emptySlots();
        if (empties.isEmpty()) { hs.pendingDraw = 0; return; }

        int toFill = Math.min(hs.pendingDraw, empties.size());
        List<Spell> draws = drawCards(w, toFill); // draw from TC first, fallback to main
        for (int i = 0; i < draws.size(); i++) {
            hs.setSlot(empties.get(i), draws.get(i));
        }
        hs.pendingDraw -= draws.size();
    }

    // Draw N cards from TC; if not enough, fallback to MAIN
    private List<Spell> drawCards(Wizard w, int n) {
        List<Spell> out = new ArrayList<>(n);
        List<Spell> tc = w.getTcDeck().getSpells();
        List<Spell> main = w.getMainDeck().getSpells();

        // Pick unique random indexes safely from each source
        int want = n;

        if (!tc.isEmpty()) {
            Set<Integer> chosen = new HashSet<>();
            while (chosen.size() < want && chosen.size() < tc.size()) {
                chosen.add(rng.nextInt(tc.size()));
            }
            for (int idx : chosen) out.add(tc.get(idx));
            want -= chosen.size();
        }
        if (want > 0 && !main.isEmpty()) {
            Set<Integer> chosen2 = new HashSet<>();
            while (chosen2.size() < want && chosen2.size() < main.size()) {
                chosen2.add(rng.nextInt(main.size()));
            }
            for (int idx : chosen2) out.add(main.get(idx));
        }
        return out;
    }

    // ---------- Discard handling ----------
    private List<Integer> askForDiscards() {
        System.out.println("Select which cards to discard (slot numbers separated by spaces). Press ENTER for none:");
        String line = sc.nextLine().trim();
        if (line.isEmpty()) return Collections.emptyList();

        String[] parts = line.split("\\s+");
        List<Integer> slots = new ArrayList<>();
        for (String p : parts) {
            try {
                int oneBased = Integer.parseInt(p);
                int zeroBased = oneBased - 1;
                if (zeroBased >= 0 && zeroBased < HAND_SIZE) slots.add(zeroBased);
            } catch (NumberFormatException ignore) {}
        }
        // De-dup + sort
        TreeSet<Integer> set = new TreeSet<>(slots);
        return new ArrayList<>(set);
    }

    private void applyDiscards(Wizard w, List<Integer> discardSlots) {
        if (discardSlots.isEmpty()) {
            System.out.println("No cards discarded. Your hand remains the same.");
            return;
        }
        HandState hs = handMap.get(w);
        for (int idx : discardSlots) {
            if (idx >= 0 && idx < hs.slots.size()) hs.setSlot(idx, null);
        }
        System.out.println("Discarded " + discardSlots.size() + " card(s). They will be replaced at the start of your next turn.");
        printHand(w);
    }

    // ---------- Display ----------
    private void printHand(Wizard w) {
        HandState hs = handMap.get(w);
        System.out.println("\nHAND FOR: " + w.getName());
        for (int i = 0; i < hs.slots.size(); i++) {
            Spell s = hs.slots.get(i);
            if (s == null) System.out.println((i + 1) + ". [EMPTY]");
            else           System.out.println((i + 1) + ". " + s.getName());
        }
    }

    // ---------- Pip roll ----------
    private void rollAndApplyPip(Wizard w) {
        System.out.println("Checking next pip (power pip chance " + w.getStatsInformation().getPowerPip() + "%)...");
        int chance = w.getStatsInformation().getPowerPip();
        int roll = rng.nextInt(100);
        String pipsByCount;
        if (roll < chance) {
            System.out.println("You got a power pip!");
            w.setPips(w.getPips() + 2);
            pipsByCount = "2|: 1 power pip" + w.getPipsByCount();
        } else {
            System.out.println("You got a regular pip!");
            w.setPips(w.getPips() + 1);
            pipsByCount = "1|: 1 regular pip" + w.getPipsByCount();
        }
        w.setPipsByCount(pipsByCount);
    }

    // ---------- File readers ----------
    private StatsInformation readStats(String path) {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            int health = Integer.parseInt(br.readLine());
            String[] f1 = br.readLine().split(",");
            int[] damage = parseIntArray(f1, 8);

            String[] f2 = br.readLine().split(",");
            int[] resist = parseIntArray(f2, 8);

            String[] f3 = br.readLine().split(",");
            int[] accuracy = parseIntArray(f3, 8);

            String[] f4 = br.readLine().split(",");
            int[] critical = parseIntArray(f4, 8);

            String[] f5 = br.readLine().split(",");
            int[] block = parseIntArray(f5, 8);

            String[] f6 = br.readLine().split(",");
            int[] pierce = parseIntArray(f6, 8);

            int stunResist = Integer.parseInt(br.readLine());
            int incoming   = Integer.parseInt(br.readLine());
            int outgoing   = Integer.parseInt(br.readLine());

            String[] f7 = br.readLine().split(",");
            int[] pipConversion = parseIntArray(f7, 7);

            int powerPip          = Integer.parseInt(br.readLine());
            int shadowPipRating   = Integer.parseInt(br.readLine());
            int archMasteryRating = Integer.parseInt(br.readLine());

            StatsInformation info = new StatsInformation(
                health, damage, resist, accuracy, critical, block, pierce,
                stunResist, incoming, outgoing, pipConversion, powerPip, shadowPipRating, archMasteryRating
            );
            System.out.println(info.toString());
            return info;
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException("Failed to read stats: " + path);
        }
    }

    // Ask how many cards to draw now (0..max). ENTER => 0
private int askImmediateDrawCount(int max) {
    if (max <= 0) return 0;
    System.out.println("Draw now? Enter number 0-" + max + " (press ENTER for 0):");
    String line = sc.nextLine().trim();
    if (line.isEmpty()) return 0;
    try {
        int n = Integer.parseInt(line);
        if (n < 0 || n > max) {
            System.out.println("Invalid number. Drawing 0 now.");
            return 0;
        }
        return n;
    } catch (NumberFormatException e) {
        System.out.println("Invalid input. Drawing 0 now.");
        return 0;
    }
}

// Fill specific empty slots with freshly drawn cards (TC first, then MAIN)
private void fillSpecificSlots(Wizard w, List<Integer> slotsToFill, int count) {
    if (count <= 0 || slotsToFill.isEmpty()) return;
    List<Spell> draws = drawCards(w, Math.min(count, slotsToFill.size()));
    HandState hs = handMap.get(w);
    for (int i = 0; i < draws.size(); i++) {
        hs.setSlot(slotsToFill.get(i), draws.get(i));
    }
}


    private List<Spell> readDeck(String path, boolean mainDeck) {
        List<Spell> spells = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            // skip header if present (simple heuristic)
            String header = br.readLine();
            if (header == null) return spells;
            if (!header.contains(",")) { /* no header, it's a card line; process it */ }
            else {
                // ok it's likely a header; continue reading rest
            }

            String line;
            while ((line = br.readLine()) != null) {
                String[] f = line.split(",");
                String name = f[0];
                if (mainDeck && "X".equals(f[1])) {
                    // X cost main-deck spell
                    int schoolPipChance = Integer.parseInt(f[2]);
                    String school = f[3];
                    spells.add(new Spell(name, 0, 14, schoolPipChance, school, ""));
                    continue;
                }
                int pips = Integer.parseInt(f[1]);
                int pipChance = Integer.parseInt(f[2]);
                String school = f[3];
                spells.add(new Spell(name, pips, pipChance, school, ""));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return spells;
    }

    private static int[] parseIntArray(String[] arr, int expected) {
        int[] out = new int[expected];
        for (int i = 0; i < Math.min(arr.length, expected); i++) {
            out[i] = Integer.parseInt(arr[i].trim());
        }
        return out;
    }

    private int safeNextInt() {
        while (true) {
            String s = sc.nextLine().trim();
            try { return Integer.parseInt(s); }
            catch (NumberFormatException e) { System.out.println("Enter a valid number:"); }
        }
    }

    // ---------- Debug ----------
    public void printWizardTeamInformation() {
        for (Wizard wizard : t1Wizards) {
            System.out.println("T1 Name: " + wizard.getName());
            System.out.println("Identity: " + wizard.getIdentity());
            System.out.println("Stats: " + wizard.getStatsInformation());
            System.out.println("Main Deck Size: " + wizard.getMainDeck().getSpells().size());
            System.out.println("TC Deck Size: " + wizard.getTcDeck().getSpells().size());
        }
        for (Wizard wizard : t2Wizards) {
            System.out.println("T2 Name: " + wizard.getName());
            System.out.println("Identity: " + wizard.getIdentity());
            System.out.println("Stats: " + wizard.getStatsInformation());
            System.out.println("Main Deck Size: " + wizard.getMainDeck().getSpells().size());
            System.out.println("TC Deck Size: " + wizard.getTcDeck().getSpells().size());
        }
    }
}

