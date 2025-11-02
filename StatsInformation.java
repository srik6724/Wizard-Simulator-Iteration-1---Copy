import java.util.HashMap;
import java.util.Map;

public class StatsInformation {
    private int health;
    private Map<String, Integer> damage = new HashMap<>();
    private Map<String, Integer> resist = new HashMap<>();
    private Map<String, Integer> accuracy = new HashMap<>();
    private Map<String, Integer> critical = new HashMap<>();
    private Map<String, Integer> block = new HashMap<>();
    private Map<String, Integer> pierce = new HashMap<>();
    //private int[] damage = new int[8];
    //private int[] resist = new int[8];
    //private int[] accuracy  = new int[8];
    //private int[] critical = new int[8];
    //private int[] block = new int[8];
    //private int[] pierce = new int[8];
    private int stunResist;
    private int incoming;
    private int outgoing;
    private Map<String, Integer> conversion = new HashMap<>();
    //private int[] conversion = new int[7];
    private int powerPip;
    private int shadowRating;
    private int archMasteryRating;
    
    public StatsInformation(int health, Map<String, Integer> damage, Map<String, Integer> resist, Map<String, Integer> accuracy, Map<String, Integer> critical, Map<String, Integer> block, Map<String, Integer> pierce, int stunResist, int incoming, int outgoing, Map<String, Integer> conversion, int powerPip, int shadowRating, int archMasteryRating) {
        this.health = health;
        this.damage = damage;
        this.resist = resist;
        this.accuracy = accuracy;
        this.critical = critical;
        this.block = block;
        this.pierce = pierce;
        this.stunResist = stunResist;
        this.incoming = incoming;
        this.outgoing = outgoing;
        this.conversion = conversion;
        this.powerPip = powerPip;
        this.shadowRating = shadowRating;
        this.archMasteryRating = archMasteryRating;
    }

    public int getHealth() {
        return health;
    }

    public void setHealth(int health) {
        this.health = health;
    }

    public Map<String, Integer> getDamage() {
        return damage;
    }

    public Map<String, Integer> getResist() {
        return resist;
    }

    public Map<String, Integer> getAccuracy() {
        return accuracy;
    }

    public Map<String, Integer> getCritical() {
        return critical;
    }

    public Map<String, Integer> getBlock() {
        return block;
    }

    public Map<String, Integer> getPierce() {
        return pierce;
    }

    public int getStunResist() {
        return stunResist;
    }

    public int getIncoming() {
        return incoming;
    }

    public int getOutgoing() {
        return outgoing;
    }

    public Map<String, Integer> getPipConversion() {
        return conversion;
    }

    public int getPowerPip() {
        return powerPip;
    }

    public int getShadowRating() {
        return shadowRating;
    }

    public int getArchMasteryRating() {
        return archMasteryRating;
    }

    public String toString() {
        return "[" + damage.get("balance") + "]\n" 
        + "[" + resist.get("fire") + "," + resist.get("ice") + "," + resist.get("storm") + "," + resist.get("myth") + "," + resist.get("life")
        + "," + resist.get("death") + "," + resist.get("balance") + "," + resist.get("shadow") + "]\n" 
        + "[" + accuracy.get("balance") + "]\n" 
        + "[" + critical.get("balance") + "]\n" 
        +  "[" + block.get("fire") + "," + block.get("ice") + "," + block.get("storm") + "," + block.get("myth") + "," + block.get("life")
        + "," + block.get("death") + "," + block.get("balance") + "," + block.get("shadow") + "]\n" 
        + "[" + pierce.get("balance") + "]"
        + stunResist + "," + incoming + "," + outgoing + "," + conversion.get("balance") + powerPip 
        + "," + shadowRating + "," + archMasteryRating;
    }
    
}
