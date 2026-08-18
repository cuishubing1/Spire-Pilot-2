"""Regression tests for native save/load behavior."""

from conftest import Game


def test_load_map_save_does_not_retrigger_neow(tmp_path):
    save_path = tmp_path / "map_select.save"

    game = Game()
    try:
        state = game.start(seed="sl1")
        state = game.skip_neow(state)
        assert state["decision"] == "map_select"

        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["type"] == "save_result"
        assert save_result["success"] is True
    finally:
        game.close()

    game = Game()
    try:
        state = game.send({"cmd": "load_save", "path": str(save_path)})
        assert state["decision"] == "map_select"
    finally:
        game.close()


def test_load_pre_neow_save_preserves_neow_choice(tmp_path):
    save_path = tmp_path / "pre_neow.save"

    game = Game()
    try:
        state = game.start(seed="sl2")
        assert state["decision"] == "event_choice"

        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["type"] == "save_result"
        assert save_result["success"] is True
    finally:
        game.close()

    game = Game()
    try:
        state = game.send({"cmd": "load_save", "path": str(save_path)})
        assert state["decision"] == "event_choice"
    finally:
        game.close()


def test_reload_save_reenters_same_combat_in_one_process(tmp_path):
    save_path = tmp_path / "combat_entrance.save"
    game = Game()
    try:
        state = game.skip_neow(game.start(seed="reload-combat-v1"))
        assert state["decision"] == "map_select"
        choice = next(value for value in state["choices"] if value["type"] == "Monster")
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True

        first = game.act("select_map_node", col=choice["col"], row=choice["row"])
        assert first["decision"] == "combat_play"
        restored = game.send({"cmd": "reload_save", "path": str(save_path)})
        assert restored["decision"] == "map_select"
        second = game.act("select_map_node", col=choice["col"], row=choice["row"])
        assert second == first
    finally:
        game.close()


def test_cached_batch_restore_replays_combat_prefix_exactly(tmp_path):
    save_path = tmp_path / "cached-combat-entrance.save"
    game = Game()
    try:
        state = game.skip_neow(game.start(seed="cached-combat-v1"))
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True
        cache_result = game.send({
            "cmd": "cache_save", "name": "test-entrance", "path": str(save_path)
        })
        assert cache_result["type"] == "ok"

        entry = {
            "cmd": "enter_room",
            "type": "combat",
            "encounter": "SHRINKER_BEETLE_WEAK",
        }
        root = game.send(entry)
        playable = next(
            card for card in root["hand"]
            if card.get("can_play") and card.get("type") == "Attack"
        )
        action_args = {"card_index": playable["index"]}
        if playable.get("target_type") == "AnyEnemy":
            action_args["target_index"] = root["enemies"][0]["index"]
        expected = game.act("play_card", **action_args)

        actual = game.send({
            "cmd": "restore_combat",
            "cache": "test-entrance",
            "entry": entry,
            "prefix": [{"action": "play_card", "args": action_args}],
        })
        assert actual == expected
    finally:
        game.close()


def test_compact_cached_restore_matches_legacy_projection(tmp_path):
    save_path = tmp_path / "compact-cached-combat-entrance.save"
    game = Game()
    try:
        state = game.skip_neow(game.start(seed="compact-cached-combat-v1"))
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True
        cache_result = game.send({
            "cmd": "cache_save", "name": "compact-test-entrance", "path": str(save_path)
        })
        assert cache_result["type"] == "ok"

        entry = {
            "cmd": "enter_room",
            "type": "combat",
            "encounter": "SHRINKER_BEETLE_WEAK",
        }
        root = game.send(entry)
        playable = next(
            card for card in root["hand"]
            if card.get("can_play") and card.get("type") == "Attack"
        )
        action_args = {"card_index": playable["index"]}
        if playable.get("target_type") == "AnyEnemy":
            action_args["target_index"] = root["enemies"][0]["index"]
        prefix = [
            {"action": "play_card", "args": action_args},
            {"action": "end_turn", "args": {}},
        ]
        request = {
            "cmd": "restore_combat",
            "cache": "compact-test-entrance",
            "entry": entry,
            "prefix": prefix,
        }

        legacy = game.send(request)
        compact = game.send({**request, "prefix_projection": "compact"})
        assert compact == legacy
    finally:
        game.close()


def test_compact_cached_restore_suffix_matches_stepwise_actions(tmp_path):
    save_path = tmp_path / "compact-suffix-combat-entrance.save"
    game = Game()
    try:
        state = game.skip_neow(game.start(seed="compact-suffix-v1"))
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True
        cache_result = game.send({
            "cmd": "cache_save", "name": "compact-suffix-entrance", "path": str(save_path)
        })
        assert cache_result["type"] == "ok"

        entry = {
            "cmd": "enter_room",
            "type": "combat",
            "encounter": "SHRINKER_BEETLE_WEAK",
        }
        root = game.send(entry)
        playable = next(
            card for card in root["hand"]
            if card.get("can_play") and card.get("type") == "Attack"
        )
        action_args = {"card_index": playable["index"]}
        if playable.get("target_type") == "AnyEnemy":
            action_args["target_index"] = root["enemies"][0]["index"]
        suffix_commands = [
            {"cmd": "action", "action": "play_card", "args": action_args},
            {"cmd": "action", "action": "end_turn", "args": {}},
        ]

        game.send({
            "cmd": "restore_combat",
            "cache": "compact-suffix-entrance",
            "entry": entry,
            "prefix": [],
        })
        expected = None
        for command in suffix_commands:
            expected = game.send(command)

        actual = game.send({
            "cmd": "restore_combat",
            "cache": "compact-suffix-entrance",
            "entry": entry,
            "prefix": [],
            "suffix": [
                {"action": command["action"], "args": command["args"]}
                for command in suffix_commands
            ],
            "prefix_projection": "compact",
        })
        assert actual == expected
    finally:
        game.close()


def test_prepared_save_reuse_is_stable_across_mutating_combat_restores(tmp_path):
    save_path = tmp_path / "prepared-reuse-combat-entrance.save"
    game = Game()
    try:
        state = game.skip_neow(game.start(seed="prepared-reuse-v1"))
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True
        cache_result = game.send({
            "cmd": "cache_save", "name": "prepared-reuse-entrance", "path": str(save_path)
        })
        assert cache_result["type"] == "ok"
        assert cache_result["prepared"] is True

        entry = {
            "cmd": "enter_room",
            "type": "combat",
            "encounter": "SHRINKER_BEETLE_WEAK",
        }
        root = game.send(entry)
        playable = next(
            card for card in root["hand"]
            if card.get("can_play") and card.get("type") == "Attack"
        )
        action_args = {"card_index": playable["index"]}
        if playable.get("target_type") == "AnyEnemy":
            action_args["target_index"] = root["enemies"][0]["index"]
        prefix = [
            {"action": "play_card", "args": action_args},
            {"action": "end_turn", "args": {}},
        ]
        base_request = {
            "cmd": "restore_combat",
            "cache": "prepared-reuse-entrance",
            "entry": entry,
            "prefix_projection": "compact",
        }
        expected_root = game.send({**base_request, "prefix": []})
        expected_after = game.send({**base_request, "prefix": prefix})

        for _ in range(20):
            actual_after = game.send({
                **base_request,
                "prefix": prefix,
                "reuse_prepared_save": True,
            })
            assert actual_after == expected_after
            actual_root = game.send({
                **base_request,
                "prefix": [],
                "reuse_prepared_save": True,
            })
            assert actual_root == expected_root
    finally:
        game.close()


def test_prepared_save_map_entry_does_not_drift_floor_counter(tmp_path):
    save_path = tmp_path / "prepared-map-entry.save"
    game = Game()
    try:
        map_state = game.skip_neow(game.start(seed="prepared-map-floor-v1"))
        choice = next(row for row in map_state["choices"] if row["type"] == "Monster")
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True
        cache_result = game.send({
            "cmd": "cache_save", "name": "prepared-map-entry", "path": str(save_path)
        })
        assert cache_result["prepared"] is True
        assert cache_result["prepared_packet_bytes"] > 0

        request = {
            "cmd": "restore_combat",
            "cache": "prepared-map-entry",
            "entry": {
                "cmd": "action",
                "action": "select_map_node",
                "args": {"col": choice["col"], "row": choice["row"]},
            },
            "prefix": [],
            "prefix_projection": "compact",
        }
        expected = game.send(request)
        assert expected["decision"] == "combat_play"
        expected_floor = expected["context"]["total_floor"]

        for _ in range(20):
            actual = game.send({**request, "reuse_prepared_save": True})
            assert actual == expected
            assert actual["context"]["total_floor"] == expected_floor
    finally:
        game.close()


def test_cached_batch_restore_can_report_internal_profile(tmp_path):
    save_path = tmp_path / "profiled-combat-entrance.save"
    game = Game()
    try:
        state = game.skip_neow(game.start(seed="profiled-combat-v1"))
        save_result = game.send({"cmd": "write_continue_save", "path": str(save_path)})
        assert save_result["success"] is True
        cache_result = game.send({
            "cmd": "cache_save", "name": "profiled-entrance", "path": str(save_path)
        })
        assert cache_result["type"] == "ok"

        entry = {
            "cmd": "enter_room",
            "type": "combat",
            "encounter": "SHRINKER_BEETLE_WEAK",
        }
        restored = game.send({
            "cmd": "restore_combat",
            "cache": "profiled-entrance",
            "entry": entry,
            "prefix": [],
            "profile": True,
        })
        assert restored["decision"] == "combat_play"
        profile = restored["_profile_ms"]
        assert set(profile) == {
            "cleanup",
            "load_save",
            "enter_combat",
            "prefix_replay",
            "set_draw_order",
            "suffix_replay",
            "final_projection",
            "server_pre_serialize_total",
        }
        assert all(value >= 0 for value in profile.values())
        assert profile["server_pre_serialize_total"] >= sum(
            profile[key]
            for key in (
                "cleanup",
                "load_save",
                "enter_combat",
                "prefix_replay",
                "set_draw_order",
                "suffix_replay",
                "final_projection",
            )
        ) - 0.1
    finally:
        game.close()
