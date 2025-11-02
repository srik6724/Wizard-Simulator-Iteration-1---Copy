public class Aura {
        private String name;
        private String description;
        private int rounds;
        private int tick;

        public Aura(String name, String description, int tick, int rounds) {
            this.name = name;
            this.description = description;
            this.tick = tick;
            this.rounds = rounds;
        }

        public String getDescription() {
            return description;
        }

        public int getRounds() {
            return rounds;
        }

        public int getTick() {
            return tick;
        }

        public void setTick(int tick) {
            this.tick = tick;
        }

        public String getName() {
            return name;
        }
}
