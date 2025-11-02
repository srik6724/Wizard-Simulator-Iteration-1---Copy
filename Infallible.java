public class Infallible {
    private String name;
    private int pierce;
    private int tick;
    private int roundNo;

    public Infallible(String name, int pierce, int tick, int roundNo) {
        this.name = name;
        this.pierce = pierce;
        this.tick = tick;
        this.roundNo = roundNo;
    }

    public String getName() {
        return name;
    }

    public int getPierce() {
        return pierce;
    }

    public int getTick() {
        return tick;
    }

    public int getRoundNo() {
        return roundNo;
    }
}
