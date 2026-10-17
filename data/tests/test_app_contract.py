"""juniper-chat-v1 carries Juniper App's tool protocol objects losslessly (P17, protocol layer)."""

import json
from pathlib import Path

import pytest

from j1data import jtok
from j1data.protocol import Message, parse_generation, serialize

CONTRACT = json.loads((Path(__file__).parent / "fixtures" / "juniper_app_contract.json").read_text())


@pytest.fixture(scope="module")
def tok():
    texts = [json.dumps(t) for t in CONTRACT["tools"]] * 20
    return jtok.train(
        jtok.CandidateSpec("t", 512, limit_alphabet=200, min_frequency=1, stem_alphabet=False), texts
    )


@pytest.mark.parametrize("tool", CONTRACT["tools"], ids=lambda t: t["name"])
def test_every_app_tool_call_round_trips(tok, tool):
    call = json.dumps({"name": tool["name"], "arguments": tool["example"]})
    enc = serialize(tok, [Message("assistant", call, "tool_call")])
    generated = enc.ids[1:]  # drop the host-supplied <|assistant|>
    parsed = parse_generation(tok, generated)
    assert parsed.stop == "call"
    assert json.loads(parsed.messages[0].content) == {"name": tool["name"], "arguments": tool["example"]}


@pytest.mark.parametrize("status", CONTRACT["statuses"])
def test_every_host_result_status_is_a_valid_tool_message(tok, status):
    body = {
        "protocolVersion": CONTRACT["protocolVersion"],
        "callId": "call-1",
        "name": "calculator.evaluate",
        "status": status,
        "result": {"value": 389.0} if status == "success" else None,
        "error": None if status == "success" else {"code": status, "message": "host message"},
    }
    ids = serialize(tok, [Message("tool", json.dumps(body))]).ids
    assert ids[0] == jtok.control_id("<|tool|>") and ids[-1] == jtok.control_id("<|end|>")
    assert json.loads(tok.decode(ids[1:-1])) == body


def test_tool_catalog_as_developer_message_has_no_control_ids(tok):
    catalog = json.dumps(
        [{k: t[k] for k in ("name", "description", "schema")} for t in CONTRACT["tools"]], ensure_ascii=False
    )
    ids = serialize(tok, [Message("developer", catalog)]).ids
    assert [i for i in ids if i < jtok.N_CONTROL] == [
        jtok.control_id("<|developer|>"),
        jtok.control_id("<|end|>"),
    ]


def test_wrong_protocol_version_is_rejected():
    from j1data.protocol import ProtocolError

    with pytest.raises(ProtocolError):
        Message("tool", json.dumps({"protocolVersion": "juniper-tool-protocol-v0", "status": "success"}))
