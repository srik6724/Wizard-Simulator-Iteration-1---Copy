public class Overtime {
    private String name;
    private int[] damagePerRound;
    private int tick;
    private int rounds;
    private String pierceSchool;
    private String school;

    public Overtime(String name, int[] damagePerRound, int tick, int rounds, String pierceSchool, String school) {
        this.name = name;
        this.damagePerRound = damagePerRound;
        this.tick = tick;
        this.rounds = rounds;
        this.pierceSchool = pierceSchool;
        this.school = school;
    }

    public String getPierceSchool() {
        return pierceSchool;
    }

    public String getSchool() {
        return school;
    }

    public String getName() {
        return name;
    }

    public int[] getDamagePerRound() {
        return damagePerRound;
    }

    public int getTick() {
        return tick;
    }

    public int getRounds() {
        return rounds;
    }
}
