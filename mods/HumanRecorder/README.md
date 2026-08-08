# HumanRecorder

Pure observation mod for Slay the Spire 2 v0.107.1 (Steam build 23811903). It records
player-visible state immediately before semantic decisions; it does not modify gameplay.

Version 0.4.1-internal keeps one append-only file for a logical climb, assigns `attempt_id`
after reloads, and emits explicit `rollback` or quarantined `resume_unmatched` records.
It also fingerprints loaded content mods, tags entity provenance, and records a
positive-HP `RunManager.OnEnded` terminal as victory. `run_start.run_context` stores
audit-only run provenance including the original string seed, numeric seed, characters,
ascension, game mode, acts, modifiers, badges, save mode and daily-run timestamp. Seed
metadata is deliberately kept outside the player-visible observation and transition view.

Version 0.4.1-internal follows the engine's model/save representation. Card enchantments and
afflictions are nullable child objects of their owning card. Card costs retain their
native local modifiers and expiration rules. Relics expose separate stack, status,
visible-counter, native `SavedProperties`, and versioned turn/combat runtime state.
Each decision also carries an `audit_state` with engine-native card/relic properties
and ordered combat piles; the importer never copies that audit-only object into the
training observation. Observed `RelicModel.Flash` calls are emitted as audit-only
`relic_trigger_observed` events rather than being presented as guaranteed effects.

The recorder captures committed single-player decisions for map travel, combat
card play and end turn, potion use/discard, events, rest sites, shops, card and
bundle selection, reward claiming/proceeding, and treasure opening/selecting/
skipping, including the dedicated boss-relic selection screen. It intentionally
does not record hover, inspection, or other tentative UI input.

Combat observations include resolved visible attack-intent damage (including
specialized attacks such as `DeathBlow`), player-visible intent label variables,
player and
enemy powers, order-free visible draw/discard/exhaust contents, resolved card
dynamic values and per-target damage, potion targeting, and character-specific
resources. Hidden draw order, pre-open treasure contents, and combat RNG are never exported.
Actual engine heal requests are emitted as audit-only `engine_event` records after
the preceding player decision; they are not added to the pre-decision agent view.

The staged install artifact is written to `dist/HumanRecorder` by `dotnet build`.
See the repository root README for build, install, capture, seal and import commands.
