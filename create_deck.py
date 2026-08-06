import re
import sys
import pandas as pd


def _normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


# Cards not yet in the database — map by normalized name to a substitute card name
SUBSTITUTIONS = {
    'specialredcard': 'Judge',
}

# TCG type symbols differ from English names (Fire=R, Fighting=F, etc.)
_ENERGY_SYMBOL = {
    'grass': 'G', 'fire': 'R', 'water': 'W', 'lightning': 'L',
    'psychic': 'P', 'fighting': 'F', 'darkness': 'D', 'dark': 'D',
    'metal': 'M', 'steel': 'M', 'dragon': 'N', 'colorless': 'C', 'fairy': 'Y',
}

def _to_basic_energy_name(name):
    """Convert e.g. 'Psychic Energy' -> 'Basic {P} Energy', or None if not an energy."""
    m = re.match(r'^(.+?)\s+Energy$', name, re.IGNORECASE)
    if m:
        symbol = _ENERGY_SYMBOL.get(m.group(1).lower())
        if symbol:
            return f"Basic {{{symbol}}} Energy"
    return None


def load_card_list(excel_path="Card_List.xlsx"):
    df = pd.read_excel(excel_path, header=None)
    df = df.dropna(subset=[0])
    df[0] = df[0].astype(int)
    df[3] = pd.to_numeric(df[3], errors='coerce')

    by_set = {}
    for _, row in df.iterrows():
        if pd.notna(row[3]):
            key = (str(row[2]).strip().upper(), int(row[3]))
            by_set[key] = int(row[0])

    # first match wins so more specific entries aren't overwritten
    by_name = {}
    for _, row in df.iterrows():
        key = _normalize(str(row[1]))
        if key not in by_name:
            by_name[key] = int(row[0])

    return by_set, by_name


def parse_import(import_path):
    entries = []
    card_re = re.compile(r'^(\d+)\s+(.+?)\s+([A-Za-z0-9]+)\s+(\d+)\s*$')
    with open(import_path, encoding='utf-8') as f:
        for line in f:
            m = card_re.match(line.strip())
            if m:
                entries.append((int(m.group(1)), m.group(2).strip(),
                                 m.group(3).upper(), int(m.group(4))))
    return entries


def build_deck(import_path, excel_path="Card_List.xlsx"):
    by_set, by_name = load_card_list(excel_path)
    entries = parse_import(import_path)

    deck = []
    missing = []

    for count, name, expansion, number in entries:
        card_id = by_set.get((expansion, number))
        if card_id is None:
            card_id = by_name.get(_normalize(name))
            if card_id is None:
                basic = _to_basic_energy_name(name)
                if basic:
                    card_id = by_name.get(_normalize(basic))
                    if card_id is not None:
                        print(f"[energy match] {name} -> {basic} -> {card_id}", file=sys.stderr)
            if card_id is None:
                sub = SUBSTITUTIONS.get(_normalize(name))
                if sub:
                    card_id = by_name.get(_normalize(sub))
                    if card_id is not None:
                        print(f"[substituted] {name} -> {sub} -> {card_id}", file=sys.stderr)
            if card_id is not None and _to_basic_energy_name(name) is None and _normalize(name) not in SUBSTITUTIONS:
                print(f"[name match] {name} {expansion} {number} -> {card_id}", file=sys.stderr)
            elif card_id is None:
                print(f"[NOT FOUND]  {name} {expansion} {number}", file=sys.stderr)
                missing.append(f"{name} {expansion} {number}")
                card_id = 0
        deck.extend([card_id] * count)

    return deck, missing


if __name__ == "__main__":
    import_path = sys.argv[1] if len(sys.argv) > 1 else "sample_import"
    excel_path  = sys.argv[2] if len(sys.argv) > 2 else "Card_List.xlsx"

    deck, missing = build_deck(import_path, excel_path)
    print(f"deck = {deck}")
    if len(deck) != 60:
        print(f"\n# WARNING: deck has {len(deck)} cards, expected 60.", file=sys.stderr)
    if missing:
        print(f"\n# Replace the {len(missing)} 0(s) above with the correct card IDs:", file=sys.stderr)
        for m in missing:
            print(f"#   {m}", file=sys.stderr)
