from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "dataset_v1.json"
SEEDS_PATH = ROOT / "config" / "seeds_v1.txt"
LOCK_PATH = ROOT / "environment.lock.json"
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
AUDIT_ROOT = DATA_ROOT / "audit"
FIXTURE_ROOT = DATA_ROOT / "fixtures"
DATASET_ROOT = DATA_ROOT / "dataset"
ARCHIVE_ROOT = ROOT / "archives"
THIRD_PARTY = ROOT / "third_party" / "sts2-cli"
DOTNET = ROOT / ".dotnet" / "dotnet.exe"
ENGINE_PROJECT = THIRD_PARTY / "src" / "Sts2Headless" / "Sts2Headless.csproj"

KNOWN_PHASES = {
    "bundle_select",
    "map_select",
    "combat_play",
    "card_select",
    "card_reward",
    "event_choice",
    "rest_site",
    "shop",
    "treasure",
    "boss_reward",
    "game_over",
}

# Exact hidden-state names are forbidden. Visible aggregate counts are allowed.
FORBIDDEN_AGENT_KEYS = {
    "audit_state",
    "checkpoint",
    "draw_order",
    "draw_pile",
    "future_encounter",
    "future_event",
    "random_state",
    "rng",
    "rng_state",
    "seed",
}
