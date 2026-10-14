"""JSON decoding contracts for coordinator model output."""

from learning.orchestrator_policy import parse_object


def test_parses_plain_and_fenced_json_actions() -> None:
    assert parse_object('{"kind":"wait_for_event"}') == {"kind": "wait_for_event"}
    assert parse_object('```json\n{"kind":"finish"}\n```') == {"kind": "finish"}


def test_rejects_malformed_or_non_action_json() -> None:
    assert parse_object("not json") is None
    assert parse_object('[{"kind":"finish"}]') is None
    assert parse_object('{"message":"missing kind"}') is None

