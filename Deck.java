import java.util.ArrayList;
import java.util.List;

public class Deck {
    private List<Spell> spells = new ArrayList<>();
    private List<String> spellNames = new ArrayList<>(); 

    public Deck(List<Spell> spells) {
        this.spells = spells;
        for(Spell spell: spells) {
            spellNames.add(spell.getName()); 
        }
    }

    public List<Spell> getSpells() {
        return spells;
    }

    public List<String> getSpellNames() {
        return spellNames;
    }
}
