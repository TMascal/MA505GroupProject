# Timothy Mascal
# April 1st, 2026
# Perform Data Labeling by finding key terms within an excel sheet
# Created with the aid of claude code.

# Concept
# Input: CSV File of All Aircraft Accidents in History
# Output: A pandas dataframe that can classify each accident by cause

import re

import pandas as pd


class AccidentCauseClassifier:
    """Classifies accident summaries into cause categories using keyword matching."""

    STOPWORDS = {
        "the", "a", "an", "and", "of", "in", "was", "to", "by", "at", "on",
        "it", "its", "that", "this", "with", "from", "for", "as", "or", "be",
        "but", "not", "into", "after", "when", "had", "were", "is", "are",
        "has", "have", "he", "she", "they", "who", "which", "while", "all",
        "also", "about", "one", "two", "three", "four", "five", "been", "no",
        "their", "during", "near", "over", "under", "than", "more", "other",
        "both", "then", "there", "would", "could", "upon", "out", "up", "down",
        "an", "due", "may", "caused", "shortly", "flight", "aircraft", "plane",
        "airplane", "crew",
    }

    # Keywords for each cause, checked in priority order.
    # The first cause whose keywords match wins.
    CAUSE_RULES: list[tuple[str, list[str]]] = [
        ("sabotage",      ["hijacked", "hijackers", "hijacker", "explosive", "bomb", "dynamite",
                           "sabotage", "suicide", "stolen", "grenade"]),
        ("shot_down",     ["shot down", "shot by", "enemy fire", "anti-aircraft", "shot by fighter", "fighter fire", "gunfire", "bullet"]),
        ("fire",          ["fire", "flames", "burned", "burnt", "exploded", "explosion"]),
        ("fuel",          ["fuel exhaustion", "fuel starvation", "ran out of fuel", "low fuel", "out of fuel"]),
        ("mechanical",    ["engine failure", "engine failed", "mechanical failure",
                           "engine", "engines", "failure", "failed", "power loss", "power failure",
                           "broken", "malfunctioning", "control problems",
                           "landing gear", "gear", "wing", "fatigue", "rudder", "elevator", "icing"]),
        ("weather",       ["weather", "fog", "rain", "visibility", "adverse",
                           "conditions", "vfr", "thunderstorm", "ice", "wind", "storm", "snow",
                           "turbulence", "windshear", "crosswind", "rainstorm", "snowstorm", "clouds"]),
        ("collision",     ["mid-air collision", "midair collision", "collided with another aircraft", "collided with another airplane", "collided"]),
        ("cfit",          ["mountain", "mountains", "mountainside", "mount", "mountainous", "terrain",
                           "trees", "struck", "hillside", "hill", "controlled flight",
                           "ocean", "lake", "river", "sea", "wooded", "jungle", "mt", "ravine",
                           "ridge", "peak", "volcano", "swamp", "channel", "foothills"]),
        ("pilot_error",   ["pilot error", "tailspin", "tail spin", "lost control", "nose dive",
                           "nosedive", "overran", "overloaded", "uncontrolled", "unable", "premature",
                           "procedure", "spin", "stall", "improperly", "inadequate", "alcohol",
                           "unqualified", "sharp turn", "turned over", "error", "improper", "procedures",
                           "stalled", "maintain", "failed to", "neglected"]),
        ("undetermined",  ["cause unknown", "cause undetermined", "undetermined", "reasons unknown",
                           "disappeared", "never found", "cause not determined", "missing", "wreckage",
                           "disintegrated"]),
    ]

    def __init__(self):
        self._patterns: list[tuple[str, re.Pattern]] = [
            (cause, re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in keywords) + r')\b', re.IGNORECASE))
            for cause, keywords in self.CAUSE_RULES
        ]

    def classify(self, summary: str) -> str:
        if not isinstance(summary, str) or not summary.strip():
            return "unknown"
        matches = [cause for cause, pattern in self._patterns if pattern.search(summary)]
        if not matches:
            return "unknown"
        # Drop undetermined if more specific causes were also matched
        if len(matches) > 1 and "undetermined" in matches:
            matches.remove("undetermined")
        return ", ".join(matches)

    def classify_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.classify)

    _MILITARY_PATTERN = re.compile(
        r'\bmilitary\b|air force|army|navy|marine corps|coast guard|royal air force|'
        r'royal australian|royal canadian|royal new zealand|luftwaffe|'
        r'air corps|national guard|air command',
        re.IGNORECASE,
    )
    _CIVILIAN_PATTERN = re.compile(r'\bprivate\b|\bair taxi\b', re.IGNORECASE)

    def classify_operator(self, operator: str) -> str:
        if not isinstance(operator, str) or not operator.strip():
            return "unknown"
        if self._MILITARY_PATTERN.search(operator):
            return "military"
        if self._CIVILIAN_PATTERN.search(operator):
            return "civilian"
        return "commercial"

    def classify_operator_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.classify_operator)

class LocationSubregionClassifier:
    """Maps Location strings to UN M49 subregions (22-subregion standard).

    Extracts the last comma-separated token from a Location string, normalises
    it, and looks it up in a comprehensive country/territory → subregion table.
    US state names and abbreviations, Canadian provinces, historical country
    names, and common misspellings are all accounted for.
    """

    # Regex to strip leading geographic qualifiers before the country name
    _STRIP_PREFIX = re.compile(
        r'^(?:off\s+(?:the\s+)?|near\s+|over\s+(?:the\s+)?|'
        r'north\s+of\s+|south\s+of\s+|(?:ne|nw|se|sw)\s+of\s+|'
        r'\d+\s+miles?\s+\w+\s+of\s+)',
        re.IGNORECASE,
    )
    # Regex to drop parenthetical qualifiers like "(Zaire)" or "(Zimbabwe)"
    _STRIP_PAREN = re.compile(r'\s*\(.*?\)\s*')

    # Single lookup table: normalised last token → UN M49 subregion
    _LOCATION_MAP: dict[str, str] = {
        # ── NORTHERN AMERICA ──────────────────────────────────────────────
        "united states": "Northern America", "usa": "Northern America",
        "canada": "Northern America", "greenland": "Northern America",
        "bermuda": "Northern America",
        # US states
        "alabama": "Northern America", "alaska": "Northern America",
        "arizona": "Northern America", "arkansas": "Northern America",
        "california": "Northern America", "colorado": "Northern America",
        "connecticut": "Northern America", "delaware": "Northern America",
        "florida": "Northern America", "georgia": "Northern America",
        "hawaii": "Northern America", "idaho": "Northern America",
        "illinois": "Northern America", "indiana": "Northern America",
        "iowa": "Northern America", "kansas": "Northern America",
        "kentucky": "Northern America", "louisiana": "Northern America",
        "maine": "Northern America", "maryland": "Northern America",
        "massachusetts": "Northern America", "michigan": "Northern America",
        "minnesota": "Northern America", "mississippi": "Northern America",
        "missouri": "Northern America", "montana": "Northern America",
        "nebraska": "Northern America", "nevada": "Northern America",
        "new hampshire": "Northern America", "new jersey": "Northern America",
        "new mexico": "Northern America", "new york": "Northern America",
        "north carolina": "Northern America", "north dakota": "Northern America",
        "ohio": "Northern America", "oklahoma": "Northern America",
        "oregon": "Northern America", "pennsylvania": "Northern America",
        "rhode island": "Northern America", "south carolina": "Northern America",
        "south dakota": "Northern America", "tennessee": "Northern America",
        "texas": "Northern America", "utah": "Northern America",
        "vermont": "Northern America", "virginia": "Northern America",
        "washington": "Northern America", "west virginia": "Northern America",
        "wisconsin": "Northern America", "wyoming": "Northern America",
        # 2-letter US state abbreviations
        "al": "Northern America", "ak": "Northern America", "az": "Northern America",
        "ar": "Northern America", "ca": "Northern America", "co": "Northern America",
        "ct": "Northern America", "de": "Northern America", "fl": "Northern America",
        "ga": "Northern America", "hi": "Northern America", "id": "Northern America",
        "il": "Northern America", "in": "Northern America", "ia": "Northern America",
        "ks": "Northern America", "ky": "Northern America", "la": "Northern America",
        "me": "Northern America", "md": "Northern America", "ma": "Northern America",
        "mi": "Northern America", "mn": "Northern America", "ms": "Northern America",
        "mo": "Northern America", "mt": "Northern America", "ne": "Northern America",
        "nv": "Northern America", "nh": "Northern America", "nj": "Northern America",
        "nm": "Northern America", "ny": "Northern America", "nc": "Northern America",
        "nd": "Northern America", "oh": "Northern America", "ok": "Northern America",
        "or": "Northern America", "pa": "Northern America", "ri": "Northern America",
        "sc": "Northern America", "sd": "Northern America", "tn": "Northern America",
        "tx": "Northern America", "ut": "Northern America", "vt": "Northern America",
        "va": "Northern America", "wa": "Northern America", "wv": "Northern America",
        "wi": "Northern America", "wy": "Northern America",
        "dc": "Northern America", "d.c.": "Northern America",
        # DC variants
        "washington d.c.": "Northern America", "washington dc": "Northern America",
        # Canadian provinces / territories
        "newfoundland": "Northern America", "ontario": "Northern America",
        "quebec": "Northern America", "british columbia": "Northern America",
        "alberta": "Northern America", "nova scotia": "Northern America",
        "new brunswick": "Northern America", "manitoba": "Northern America",
        "saskatchewan": "Northern America", "labrador": "Northern America",
        "yukon territory": "Northern America", "yukon": "Northern America",
        "british columbia canada": "Northern America",
        # US misspellings / alternate spellings
        "minnisota": "Northern America", "tennesee": "Northern America",
        "massachusett": "Northern America", "massachutes": "Northern America",
        "ilinois": "Northern America", "coloado": "Northern America",
        "deleware": "Northern America", "oklohoma": "Northern America",
        "louisana": "Northern America", "arazona": "Northern America",
        "airzona": "Northern America", "washingon": "Northern America",
        "wisconson": "Northern America", "calilfornia": "Northern America",
        "cailifornia": "Northern America", "south dekota": "Northern America",
        "alaksa": "Northern America", "alakska": "Northern America",

        # ── CARIBBEAN ─────────────────────────────────────────────────────
        "cuba": "Caribbean", "puerto rico": "Caribbean",
        "haiti": "Caribbean", "hati": "Caribbean",
        "dominican republic": "Caribbean", "domincan republic": "Caribbean",
        "trinidad": "Caribbean", "martinique": "Caribbean",
        "bahamas": "Caribbean", "virgin islands": "Caribbean",
        "u.s. virgin islands": "Caribbean", "us virgin islands": "Caribbean",
        "british virgin islands": "Caribbean", "jamaica": "Caribbean",
        "jamacia": "Caribbean", "barbados": "Caribbean",
        "antigua": "Caribbean", "guadeloupe": "Caribbean",
        "netherlands antilles": "Caribbean", "dominica": "Caribbean",
        "saint lucia": "Caribbean", "saint lucia island": "Caribbean",
        "sainte lucia island": "Caribbean", "west indies": "Caribbean",
        "turks & caicos islands": "Caribbean", "turks and caicos islands": "Caribbean",
        "leeward islands": "Caribbean", "french west indies": "Caribbean",
        "grenadines islands": "Caribbean", "great inagua": "Caribbean",
        "caribbean": "Caribbean",

        # ── CENTRAL AMERICA ───────────────────────────────────────────────
        "mexico": "Central America", "guatemala": "Central America",
        "honduras": "Central America", "nicaragua": "Central America",
        "el salvador": "Central America", "panama": "Central America",
        "costa rica": "Central America", "belize": "Central America",

        # ── SOUTH AMERICA ─────────────────────────────────────────────────
        "brazil": "South America", "colombia": "South America",
        "venezuela": "South America", "peru": "South America",
        "bolivia": "South America", "boliva": "South America",
        "argentina": "South America", "aregntina": "South America",
        "ecuador": "South America", "chile": "South America",
        "chili": "South America", "uruguay": "South America",
        "guyana": "South America", "french guiana": "South America",
        "dutch guyana": "South America", "paraguay": "South America",
        "suriname": "South America", "surinam": "South America",

        # ── NORTHERN AFRICA ───────────────────────────────────────────────
        "egypt": "Northern Africa",
        "uar": "Northern Africa",   # United Arab Republic (Egypt–Syria union)
        "morocco": "Northern Africa", "morrocco": "Northern Africa",
        "morroco": "Northern Africa", "algeria": "Northern Africa",
        "algiers": "Northern Africa", "libya": "Northern Africa",
        "tunisia": "Northern Africa", "sudan": "Northern Africa",
        "mauritania": "Western Africa", "mauretania": "Western Africa",
        "canary islands": "Southern Europe",  # Spanish autonomous community

        # ── EASTERN AFRICA ────────────────────────────────────────────────
        "ethiopia": "Eastern Africa", "kenya": "Eastern Africa",
        "somalia": "Eastern Africa", "tanzania": "Eastern Africa",
        "uganda": "Eastern Africa", "rwanda": "Eastern Africa",
        "mozambique": "Eastern Africa", "mocambique": "Eastern Africa",
        "central mozambique": "Eastern Africa",
        "madagascar": "Eastern Africa", "malawi": "Eastern Africa",
        "zambia": "Eastern Africa", "zimbabwe": "Eastern Africa",
        "rhodesia": "Eastern Africa", "djibouti": "Eastern Africa",
        "djbouti": "Eastern Africa", "eritrea": "Eastern Africa",
        "comoros": "Eastern Africa", "comoro islands": "Eastern Africa",
        "comoros islands": "Eastern Africa", "south sudan": "Eastern Africa",
        "tanganyika": "Eastern Africa",
        "french somaliland": "Eastern Africa",
        "malagasy republic": "Eastern Africa",
        "reunion": "Eastern Africa",  # French island territory

        # ── MIDDLE AFRICA ─────────────────────────────────────────────────
        "angola": "Middle Africa", "congo": "Middle Africa",
        "zaire": "Middle Africa", "dr congo": "Middle Africa",
        "cameroon": "Middle Africa", "cameroons": "Middle Africa",
        "french cameroons": "Middle Africa",
        "central african republic": "Middle Africa",
        "chad": "Middle Africa", "equatorial guinea": "Middle Africa",
        "gabon": "Middle Africa", "belgian congo": "Middle Africa",
        "belgium congo": "Middle Africa",
        "french equatorial africa": "Middle Africa",
        "french equitorial africa": "Middle Africa",
        # DRC truncation variants from CSV encoding
        "democratirepublicongo": "Middle Africa",
        "democratirepublicogo": "Middle Africa",
        "democratirepubliof congo": "Middle Africa",
        "democtratirepublicongo": "Middle Africa",

        # ── SOUTHERN AFRICA ───────────────────────────────────────────────
        "south africa": "Southern Africa", "namibia": "Southern Africa",
        "botswana": "Southern Africa", "lesotho": "Southern Africa",

        # ── WESTERN AFRICA ────────────────────────────────────────────────
        "nigeria": "Western Africa", "senegal": "Western Africa",
        "gambia": "Western Africa", "ghana": "Western Africa",
        "ivory coast": "Western Africa", "guinea": "Western Africa",
        "liberia": "Western Africa", "mali": "Western Africa",
        "sierra leone": "Western Africa", "sierre leone": "Western Africa",
        "niger": "Western Africa", "benin": "Western Africa",
        "cape verde": "Western Africa", "cape verde islands": "Western Africa",
        "burkina faso": "Western Africa", "upper volta": "Western Africa",
        "french west africa": "Western Africa", "togo": "Western Africa",

        # ── CENTRAL ASIA ──────────────────────────────────────────────────
        "kazakhstan": "Central Asia", "uzbekistan": "Central Asia",
        "kyrgyzstan": "Central Asia", "kirghizia": "Central Asia",
        "tajikistan": "Central Asia", "turkmenistan": "Central Asia",

        # ── EASTERN ASIA ──────────────────────────────────────────────────
        "china": "Eastern Asia", "japan": "Eastern Asia",
        "taiwan": "Eastern Asia", "hong kong": "Eastern Asia",
        "south korea": "Eastern Asia", "north korea": "Eastern Asia",
        "mongolia": "Eastern Asia", "okinawa": "Eastern Asia",
        "formosa strait": "Eastern Asia",

        # ── SOUTH-EASTERN ASIA ────────────────────────────────────────────
        "philippines": "South-eastern Asia",
        "phillipines": "South-eastern Asia",
        "philipines": "South-eastern Asia",
        "indonesia": "South-eastern Asia",
        "inodnesia": "South-eastern Asia",
        "vietnam": "South-eastern Asia",
        "south vietnam": "South-eastern Asia",
        "thailand": "South-eastern Asia", "thiland": "South-eastern Asia",
        "malaysia": "South-eastern Asia", "laos": "South-eastern Asia",
        "cambodia": "South-eastern Asia",
        "khmer republic": "South-eastern Asia",
        "myanmar": "South-eastern Asia", "burma": "South-eastern Asia",
        "manmar": "South-eastern Asia",
        "singapore": "South-eastern Asia",
        "timor": "South-eastern Asia", "east timor": "South-eastern Asia",
        "brunei": "South-eastern Asia", "borneo": "South-eastern Asia",
        "sarawak": "South-eastern Asia", "malaya": "South-eastern Asia",
        "french indo-china": "South-eastern Asia",
        "netherlands indies": "South-eastern Asia",

        # ── SOUTHERN ASIA ─────────────────────────────────────────────────
        "india": "Southern Asia", "pakistan": "Southern Asia",
        "west pakistan": "Southern Asia",
        "bangladesh": "Southern Asia", "baangladesh": "Southern Asia",
        "nepal": "Southern Asia", "napal": "Southern Asia",
        "afghanistan": "Southern Asia", "sri lanka": "Southern Asia",
        "bhutan": "Southern Asia",

        # ── WESTERN ASIA ──────────────────────────────────────────────────
        "iran": "Western Asia", "turkey": "Western Asia",
        "iraq": "Western Asia", "syria": "Western Asia",
        "israel": "Western Asia", "jordan": "Western Asia",
        "saudi arabia": "Western Asia", "lebanon": "Western Asia",
        "kuwait": "Western Asia",
        "united arab emirates": "Western Asia", "uae": "Western Asia",
        "qatar": "Western Asia", "bahrain": "Western Asia",
        "oman": "Western Asia", "yemen": "Western Asia",
        "south yemen": "Western Asia",
        "azerbaijan": "Western Asia", "armenia": "Western Asia",
        "cyprus": "Western Asia",
        "republic of georgia": "Western Asia",
        "republics of georgia": "Western Asia",
        "republiof georgia": "Western Asia",

        # ── EASTERN EUROPE ────────────────────────────────────────────────
        "russia": "Eastern Europe", "ussr": "Eastern Europe",
        "soviet union": "Eastern Europe", "russian": "Eastern Europe",
        "poland": "Eastern Europe", "czech republic": "Eastern Europe",
        "czechoslovakia": "Eastern Europe", "slovakia": "Eastern Europe",
        "hungary": "Eastern Europe", "hunary": "Eastern Europe",
        "romania": "Eastern Europe", "romainia": "Eastern Europe",
        "bulgaria": "Eastern Europe", "bulgeria": "Eastern Europe",
        "bugaria": "Eastern Europe",
        "ukraine": "Eastern Europe", "moldova": "Eastern Europe",
        "east germany": "Eastern Europe", "chechnya": "Eastern Europe",

        # ── NORTHERN EUROPE ───────────────────────────────────────────────
        "england": "Northern Europe", "scotland": "Northern Europe",
        "wales": "Northern Europe", "northern ireland": "Northern Europe",
        "united kingdom": "Northern Europe", "uk": "Northern Europe",
        "ireland": "Northern Europe", "norway": "Northern Europe",
        "sweden": "Northern Europe", "denmark": "Northern Europe",
        "finland": "Northern Europe", "iceland": "Northern Europe",
        "estonia": "Northern Europe", "latvia": "Northern Europe",
        "lithuania": "Northern Europe", "jersey": "Northern Europe",
        "isle of man": "Northern Europe", "islay island": "Northern Europe",
        "guernsey": "Northern Europe",
        "azores": "Southern Europe",   # Portuguese autonomous region

        # ── SOUTHERN EUROPE ───────────────────────────────────────────────
        "spain": "Southern Europe", "italy": "Southern Europe",
        "greece": "Southern Europe", "portugal": "Southern Europe",
        "yugoslavia": "Southern Europe", "yugosalvia": "Southern Europe",
        "malta": "Southern Europe", "albania": "Southern Europe",
        "croatia": "Southern Europe", "slovenia": "Southern Europe",
        "bosnia": "Southern Europe",
        "bosnia herzegovina": "Southern Europe",
        "bosnia-herzegovina": "Southern Europe",
        "serbia": "Southern Europe", "macedonia": "Southern Europe",
        "kosovo": "Southern Europe",

        # ── WESTERN EUROPE ────────────────────────────────────────────────
        "france": "Western Europe", "germany": "Western Europe",
        "west germany": "Western Europe", "belgium": "Western Europe",
        "netherlands": "Western Europe", "the netherlands": "Western Europe",
        "switzerland": "Western Europe", "austria": "Western Europe",
        "luxembourg": "Western Europe",

        # ── AUSTRALIA AND NEW ZEALAND ─────────────────────────────────────
        "australia": "Australia and New Zealand",
        "new zealand": "Australia and New Zealand",
        "tasmania": "Australia and New Zealand",

        # ── MELANESIA ─────────────────────────────────────────────────────
        "papua new guinea": "Melanesia", "new guinea": "Melanesia",
        "fiji": "Melanesia", "vanuatu": "Melanesia",
        "solomon islands": "Melanesia",

        # ── MICRONESIA ────────────────────────────────────────────────────
        "guam": "Micronesia", "mariana islands": "Micronesia",
        "marshall islands": "Micronesia", "wake island": "Micronesia",

        # ── POLYNESIA ─────────────────────────────────────────────────────
        "french polynesia": "Polynesia", "tahiti": "Polynesia",
        "american samoa": "Polynesia", "u.s. samoa": "Polynesia",
        "western samoa": "Polynesia", "west samoa": "Polynesia",
        "samoa": "Polynesia", "cook islands": "Polynesia",
    }

    def _extract_token(self, location: str) -> str:
        """Return the normalised country token from a location string."""
        if not isinstance(location, str) or not location.strip():
            return ""
        token = location.split(",")[-1].strip()
        token = self._STRIP_PAREN.sub("", token).strip()
        token = token.rstrip(".")
        token = token.lower().strip()
        token = self._STRIP_PREFIX.sub("", token).strip()
        return token

    def classify_location(self, location: str) -> str:
        token = self._extract_token(location)
        if not token:
            return "unknown"
        return self._LOCATION_MAP.get(token, "unknown")

    def classify_location_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.classify_location)


if __name__ == "__main__":
    df = pd.read_csv("data/Airplane_Crashes_and_Fatalities_Since_1908.csv")
    print(f"Loaded {len(df)} rows.\n")

    classifier = AccidentCauseClassifier()
    loc_classifier = LocationSubregionClassifier()

    aboard      = pd.to_numeric(df["Aboard"],      errors="coerce")
    fatalities  = pd.to_numeric(df["Fatalities"],  errors="coerce")
    ground      = pd.to_numeric(df["Ground"],      errors="coerce").fillna(0)
    fatality_rate = (fatalities + ground) / aboard  # NaN when Aboard is 0 or missing

    labeled = pd.DataFrame({
        "Date":          df["Date"],
        "Time":          df["Time"],
        "Location":      df["Location"],
        "Subregion":     loc_classifier.classify_location_series(df["Location"]),
        "Operator":      df["Operator"],
        "FlightType":    classifier.classify_operator_series(df["Operator"]),
        "Cause":         classifier.classify_series(df["Summary"]),
        "Aboard":        aboard,
        "Fatalities":    fatalities,
        "FatalityRate":  fatality_rate,
    })

    labeled.to_csv("data/labeled_accidents.csv", index=False)
    print("Saved to data/labeled_accidents.csv\n")

    print("Cause distribution:")
    print(labeled["Cause"].value_counts().to_string())
    print()
    print("Cause ratios:")
    print(labeled["Cause"].value_counts(normalize=True).map(lambda x: f"{x:.2%}").to_string())
    print()
    print("Subregion distribution:")
    print(labeled["Subregion"].value_counts().to_string())
    print()
    print(f"Subregion unknown rate: {(labeled['Subregion'] == 'unknown').mean():.2%}")
    print()
    print("Sample:")
    print(labeled.head(10).to_string())