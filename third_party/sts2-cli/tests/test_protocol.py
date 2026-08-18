"""Protocol extensions required by the Spire Pilot 2 engine adapter."""


def test_get_state_is_non_advancing(game):
    state = game.skip_neow(game.start(seed="get-state-v1"))
    assert state["decision"] == "map_select"

    first = game.send({"cmd": "get_state"})
    second = game.send({"cmd": "get_state"})

    assert first["decision"] == "map_select"
    assert first["choices"] == state["choices"]
    assert second["choices"] == first["choices"]
    assert second["floor"] == first["floor"]
