# HumanRecorder content-state design for STS2 v0.107.1

## Static audit

The audit is derived from the pinned `sts2.dll` for build 23811903. It enumerates concrete model classes, declared state members, save attributes, gameplay hooks, and selected effect references.

| Family | Concrete classes | Mock/deprecated | Production candidates | Classes with engine-saved custom properties |
| --- | ---: | ---: | ---: | ---: |
| Cards | 584 | 7 | 577 | 5 (8 properties) |
| Enchantments | 24 | 2 | 22 | 0 |
| Afflictions | 10 | 3 | 7 | 0 |
| Relics | 297 | 1 | 296 | 40 (47 properties) |

The full machine-readable inventory is `artifacts/content-audit-v0.107.1.json`. These counts describe classes in the assembly, not necessarily content obtainable in one standard run.

## Design rules

1. A card owns at most one enchantment, so `enchantment` is a nullable child object of the card instance. Do not keep `enchantment_id` and `enchantment_amount` as parallel top-level card fields.
2. Affliction is a separate nullable child object because a card can carry an enchantment and an affliction independently.
3. Record both authoritative state and observable projections. Engine-saved properties are authoritative for audit and replay, but are not automatically safe for `agent_observation` because some can encode information not visible to the player.
4. Record causal trigger events separately from snapshots. A snapshot says what state is true; an event such as `relic_triggered` explains why the transition happened.
5. Unknown decision-relevant mutable state must lower capture quality or quarantine the sample. It must not be silently omitted.

## Card instance

Recommended raw shape:

```json
{
  "instance_id": "stable-within-episode",
  "id": "CARD...",
  "pile": "hand",
  "upgrade_level": 1,
  "floor_added": 6,
  "enchantment": {
    "id": "ENCHANTMENT...",
    "amount": 2,
    "status": "Normal",
    "display_amount": 2,
    "dynamic_vars": {},
    "runtime_state": {}
  },
  "affliction": null,
  "energy_cost": {
    "canonical": 2,
    "current": 1,
    "costs_x": false,
    "captured_x": null,
    "modifiers": [
      { "kind": "set", "value": 1, "expires": "turn_end" }
    ]
  },
  "star_cost": {
    "canonical": 0,
    "current": 0,
    "costs_x": false,
    "modifiers": []
  },
  "keywords": [],
  "tags": [],
  "dynamic_vars": {
    "base": {},
    "effective": {},
    "preview": {}
  },
  "runtime_flags": {
    "exhaust_on_next_play": false,
    "retain_this_turn": false,
    "sly_this_turn": false
  },
  "playability": {
    "can_play": true,
    "reason": null,
    "reason_source_id": null
  },
  "persistent_state": [],
  "runtime_state": []
}
```

The engine has only eight card-specific saved properties in this build: `GeneticAlgorithm.CurrentBlock`, `GeneticAlgorithm.IncreasedBlock`, `Guilty.CombatsSeen`, `MadScience.TinkerTimeRider`, `MadScience.TinkerTimeType`, `SpoilsMap.SpoilsActIndex`, `TheScythe.CurrentDamage`, and `TheScythe.IncreasedDamage`. Combat-only values on cards such as temporary costs, current dynamic variables, and conditional playability still need snapshot capture even though they are not save properties.

The current runtime object hash is not a sufficient long-lived card identity because objects can be recreated by save/load. Keep it as `engine_object_ref` for audit and assign/reconcile a recorder-owned `instance_id` across rollback branches.

## Enchantment child

All production enchantments are covered by the common fields `id`, `amount`, `status`, `display_amount`, and `dynamic_vars`. Two additional runtime values need explicit capture in this build:

- `Glam.UsedThisCombat`
- `Momentum.ExtraDamage`

`Slither.TestEnergyCostOverride` is test scaffolding and must not enter production observations; its actual effect is represented by the card's energy-cost modifiers. `Swift`, `Sown`, and `Vigorous` use the common enchantment `Status` to mark their first-use effect as consumed. `Goopy` grows its common `Amount`, so no special field is required.

## Relic instance

Relics should use a layered representation rather than a single `amount` field:

```json
{
  "instance_id": "relic:...",
  "id": "RELIC...",
  "stack_count": 1,
  "floor_added": 4,
  "status": "Normal",
  "visible_state": {
    "show_counter": true,
    "display_amount": 1,
    "is_used_up": false,
    "is_wax": false,
    "is_melted": false
  },
  "dynamic_vars": {},
  "persistent_state": [
    {
      "key": "times_used",
      "value_type": "int",
      "int_value": 2,
      "lifecycle": "run",
      "visibility": "public",
      "source": "saved_property"
    }
  ],
  "runtime_state": [
    {
      "key": "used_this_combat",
      "value_type": "bool",
      "bool_value": false,
      "lifecycle": "combat",
      "visibility": "public",
      "source": "v0.107.1_adapter"
    }
  ]
}
```

Use a tagged union for state values in Parquet: `int`, `decimal`, `bool`, `string`, `enum`, `model_id`, `int_list`, `card`, and `card_list`. Every variable also carries lifecycle (`turn`, `combat`, `room`, `act`, `run`, or `permanent`), visibility (`public` or `audit_only`), and provenance.

Relic handling falls into four groups:

1. Stateless/passive or on-pickup relics: identity plus resulting state deltas is sufficient, but a `relic_triggered` or `relic_obtained` event improves causal learning.
2. Visible counters and charges: always record `ShowCounter`, `DisplayAmount`, and `IsUsedUp`. The assembly contains 39 counter-showing relic classes and 37 `DisplayAmount` overrides.
3. Persistent run state: capture the game's `SavedProperties`/`ToSerializable().Props` representation. This covers 47 custom properties across 40 relic classes, including Silver Crucible counters, Winged Boots uses, Girya lifts, Pael's Tooth cards, and Fur Coat coordinates.
4. Transient decision state: capture a versioned adapter registry for turn/combat flags and counters not included in save properties. Examples include `ArtOfWar.AnyAttacksPlayedThisTurn`, `CentennialPuzzle.UsedThisCombat`, `RuinedHelmet.UsedThisCombat`, `RainbowRing` per-turn counters, and `Vambrace.BlockGainedThisCombat`.

Engine-saved properties are first written to `audit_state`. A reviewed projection registry decides which become public observations. This avoids leaking hidden choices or future information while retaining exact replay data.

## Trigger events

Snapshots alone cannot reliably teach whether a potion appeared because of Petrified Toad or another source. Emit structured events when possible:

```json
{
  "event_type": "relic_triggered",
  "source_instance_id": "relic:...",
  "source_id": "RELIC.PETRIFIED_TOAD",
  "trigger": "combat_start",
  "effects": [
    {
      "effect_type": "potion_procured",
      "entity_id": "POTION.POTION_SHAPED_ROCK",
      "entity_instance_id": "potion:...",
      "success": true
    }
  ]
}
```

Events are training aids and audit evidence, not replacements for before/after snapshots.

## Validation gates

- Compare every model's engine-serialized properties with recorder output.
- Maintain a v0.107.1 public-projection registry for all 40 relic classes with saved custom state and all transient-state candidates.
- Assert that enchanted cards have exactly one nullable `enchantment` child and no legacy flat enchantment fields in Dataset V2.
- Assert that all card and relic state variables use registered types, lifecycle, visibility, and provenance.
- Exercise stateful representatives at turn, combat, room, act, and run boundaries, including save/load and rollback.
- Unknown modded saved properties may be retained in audit form; unknown public runtime fields must be flagged for review before the episode is admitted to the default training partition.

## Implementation status

HumanRecorder 0.4.0 implements this representation with live schema
`human-live-0.4.0` and native-state schema `native-model-state-0.1.0`.
The public projection registry is `sts2-v0.107.1-state-projection-1`.
