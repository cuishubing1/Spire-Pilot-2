"""Developer diagnostic for the targeted boss-reward fixture."""

from sts2_dataset.constants import CONFIG_PATH
from sts2_dataset.engine import Sts2Engine
from sts2_dataset.normalize import normalize_observation
from sts2_dataset.policy import HeuristicPolicy
from sts2_dataset.smoke import _advance_to_map
from sts2_dataset.util import load_json

config = load_json(CONFIG_PATH)
run_id = "boss-probe"
engine = Sts2Engine(config, run_id)
policy = HeuristicPolicy(run_id)
try:
    state = engine.reset(seed=run_id)
    state = _advance_to_map(engine, state, config, run_id, policy)
    boss_id = engine.get_map()["boss"]["id"]
    print("boss", boss_id)
    print(engine.raw_command({
        "cmd": "set_player",
        "hp": 999,
        "max_hp": 999,
        "deck": ["POMMEL_STRIKE"] * 10,
    }).get("type"))
    state = engine.raw_command({"cmd": "enter_room", "type": "combat", "encounter": boss_id})
    for step in range(2000):
        if state.get("decision") != "combat_play":
            print("result", step, state.get("decision"), state.get("victory"))
            break
        obs = normalize_observation(state, config=config, run_id=run_id, step_id=step, audit_ref=None)
        state = engine.step(policy.choose(obs.to_dict()))
    else:
        print("timeout", state.get("decision"))
    for post_step in range(50):
        print("post", post_step, state.get("decision"), (state.get("context") or {}).get("act"))
        if state.get("decision") in {"map_select", "game_over"}:
            break
        obs = normalize_observation(state, config=config, run_id=run_id, step_id=2000 + post_step, audit_ref=None)
        state = engine.step(policy.choose(obs.to_dict()))
finally:
    engine.close()
