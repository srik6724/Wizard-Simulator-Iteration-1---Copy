import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

public class Wizard {
    private String name;
    private String identity;
    private StatsInformation statsInformation;
    private Deck mainDeck;
    private Deck tcDeck;
    private List<String> positiveCharms;
    private List<String> negativeCharms;
    private List<String> shieldsList;
    private Multimap<String, String> traps;
    private List<String> trapsList;
    private List<String> effects;
    private List<String> shadowEffects;
    private List<String> infections; 
    private List<Overtime> overtimes;
    private List<String> weaknessesList;
    private Multimap<String, String> blades;
    private Multimap<String, String> weaknesses;
    private Multimap<String, String> shields;
    private List<String> shieldsAndTraps;
    private int pips; 
    private String pipsByCount;
    private Aura aura;
    private Infallible infallible;
    private String pipsByColor;
    private int shadowGauge; 
    private boolean isStunned;
    private List<Spell> activeHand = new ArrayList<>(); 
    private Set<Integer> chosen = new HashSet<>();

    public Wizard(String name, String identity, StatsInformation statsInformation, Deck mainDeck, Deck tcDeck, int pips) {
        this.name = name;
        this.identity = identity;
        this.statsInformation = statsInformation;
        this.mainDeck = mainDeck;
        this.tcDeck = tcDeck;
        positiveCharms = new ArrayList<>(); 
        negativeCharms = new ArrayList<>();
        shieldsList = new ArrayList<>();
        traps = ArrayListMultimap.create();
        trapsList = new ArrayList<>(); 
        effects = new ArrayList<>(); 
        infections = new ArrayList<>(); 
        weaknesses = ArrayListMultimap.create();
        weaknessesList = new ArrayList<>();
        overtimes = new ArrayList<>();
        blades = ArrayListMultimap.create();
        shields = ArrayListMultimap.create();
        shieldsAndTraps = new ArrayList<>();
        shadowEffects = new ArrayList<>();
        this.pips = pips;
        shadowGauge = 0;
        isStunned = false;
    }

    public Infallible getInfallible() {
        return infallible;
    }

    public void setInfallible(Infallible infallible) {
        this.infallible = infallible;
    }

    public boolean getIsStunned() {
        return isStunned;
    }

    public void setIsStunned(boolean isStunned) {
        this.isStunned = isStunned;
    }

    public List<Overtime> getOvertimes() {
        return overtimes;
    }

    public int getShadowGauge() {
        return shadowGauge;
    }

    public void setShadowGauge(int shadowGauge) {
        this.shadowGauge = shadowGauge;
    }

    public List<Spell> getActiveHand() {
        return activeHand;
    }

    public Set<Integer> getChosen() {
        return chosen;
    }

    public List<String> getShadowEffects() {
        return shadowEffects;
    }

    public List<String> getShieldsAndTraps() {
        return shieldsAndTraps;
    }

    public Multimap<String, String> getShields() {
        return shields;
    }

    public List<String> getTrapsList() {
        return trapsList;
    }

    public List<String> getInfections() {
        return infections;
    }

    public Multimap<String, String> getWeaknesses() {
        return weaknesses;
    }

    public List<String> getWeaknessesList() {
        return weaknessesList;
    }

    public Multimap<String, String> getBlades() {
        return blades;
    }

    public Aura getAura() {
        return aura;
    }

    public void setAura(Aura aura) {
        this.aura = aura;
    }

    public void setPips(int pips) {
        this.pips = pips;
    }

    public void setPipsByCount(String pipsByCount) {
        this.pipsByCount = pipsByCount;
    }

    public String getPipsByCount(int initial) {
        String str = "";
        if(initial == 1) {
            int j = 2; 
            for(int i = 2; i <= pips; i+=2) {
                str += j + "|" + ": 1 power pip"; 
            }
            str += "1" + ": 1 regular pip";
        }
        return str;
    }

    public String getPipsByCount() {
        return pipsByCount;
    }

    public List<String> getPositiveCharms() {
        return positiveCharms;
    }

    public List<String> getNegativeCharms() {
        return negativeCharms;
    }

    public Multimap<String,String> getTraps() {
        return traps;
    }

    public List<String> getShieldsList() {
        return shieldsList;
    }

    public List<String> getEffects() {
        return effects;
    }

    public int getPips() {
        return pips;
    }

    public String getName() {
        return name;
    }

    public String getIdentity() {
        return identity;
    }

    public StatsInformation getStatsInformation() {
        return statsInformation;
    }

    public Deck getMainDeck() {
        return mainDeck;
    }

    public Deck getTcDeck() {
        return tcDeck;
    }
}
