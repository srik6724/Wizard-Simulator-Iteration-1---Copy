public class Spell {
    private String name;
    private int pips;
    private int pipChance;
    private String school;
    private String description;

    private int lowPips;
    private int maxPips;

    public Spell(String name, int pips, int pipChance, String school, String description) {
        this.name = name;
        this.pips = pips;
        this.pipChance = pipChance;
        this.school = school;
        this.description = description;
    }

     public Spell(String name, int lowPips, int maxPips, int pipChance, String school, String description) {
        this.name = name;
        this.lowPips = lowPips;
        this.maxPips = maxPips;
        this.pipChance = pipChance;
        this.school = school;
        this.description = description;
    }



    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getPips() {
        return pips;
    }

    public int getPipChance() {
        return pipChance;
    }

    public String getSchool() {
        return school;
    }

    public String getDescriptino() {
        return description;
    }

    public int getLowPips() {
        return lowPips;
    }

    public int getMaxPips() {
        return maxPips;
    }

    @Override
    public String toString() {
        String pipText;

    // If it's an X-pip spell (lowPips/maxPips set)
    if (maxPips > 0) {
        pipText = lowPips + "–" + maxPips + " pips";
    } else {
        pipText = pips + " pips";
    }

    return String.format(
        "%s [%s] - %s (Chance: %d%%)",
        name, school, pipText, pipChance, description
    );
    }
}
