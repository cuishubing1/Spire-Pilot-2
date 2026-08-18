import copy

import pytest
from jsonschema.exceptions import ValidationError

from sts2_dataset.combat_tool_cli import execute_request

from test_combat_tool import _sample, _tool


def test_execute_request_runs_typed_tool_call():
    response = execute_request(
        _tool(),
        {
            "schema_version": "combat-tool-request-0.1.0",
            "sample": _sample(),
            "directive": {"resource_policy": {"max_potion_uses": 0}},
            "top_k": 2,
        },
    )
    assert response["schema_version"] == "combat-tool-0.2.0"
    assert len(response["top_k"]) == 2
    assert response["chosen_action"]["action_type"] != "use_potion"


def test_execute_request_rejects_unversioned_request():
    request = {"sample": copy.deepcopy(_sample())}
    with pytest.raises(ValidationError):
        execute_request(_tool(), request)
