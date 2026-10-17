# `juniper-chat-v1` — Juniper 1 native conversation and tool protocol

- **Status:** implemented and tested in Juniper-1-Data (`src/j1data/protocol.py`,
  `src/j1data/jtok.py`, `tests/test_jtok_protocol.py`). It becomes frozen together with
  the tokenizer (`j1.tokenizer.v1`).
- **Teacher format:** GPT-OSS's Harmony format is *not* inherited. Teacher outputs are
  re-serialized into this protocol, and their control strings are not reused.

## 1. Control tokens (IDs fixed in every tokenizer candidate)

| ID | Token | Meaning |
|---:|---|---|
| 0 | `<\|endoftext\|>` | document separator in pretraining packs; padding ID (always loss-masked) |
| 1 | `<\|system\|>` | host instructions (highest authority) |
| 2 | `<\|developer\|>` | application / profile instructions, tool schemas |
| 3 | `<\|user\|>` | user message |
| 4 | `<\|assistant\|>` | assistant message header |
| 5 | `<\|tool\|>` | host-authored tool result |
| 6 | `<\|analysis\|>` | assistant reasoning channel |
| 7 | `<\|final\|>` | assistant user-facing answer channel |
| 8 | `<\|tool_call\|>` | assistant tool-request channel |
| 9 | `<\|end\|>` | end of a non-terminal message |
| 10 | `<\|call\|>` | end of a tool request; generation stops; the host must act |
| 11 | `<\|eot\|>` | end of the assistant turn; generation stops |
| 12–63 | `<\|reserved_NN\|>` | no meaning; never emitted, never trained |

The total vocabulary *includes* these 64 IDs and the 256 byte-fallback IDs (64–319).

## 2. Grammar

```
conversation := message*
message      := <|system|>    TEXT <|end|>
              | <|developer|> TEXT <|end|>
              | <|user|>      TEXT <|end|>
              | <|tool|>      RESULT_JSON <|end|>            (host only)
              | <|assistant|> <|analysis|>  TEXT <|end|>
              | <|assistant|> <|tool_call|> CALL_JSON <|call|>
              | <|assistant|> <|final|>     TEXT <|eot|>
CALL_JSON    := {"name": <string>, "arguments": <object>}
RESULT_JSON  := Juniper App HostToolResult (protocolVersion "juniper-tool-protocol-v1")
```

A tool request can never be mistaken for a completed turn: `<|call|>` and `<|eot|>` are
distinct IDs. A turn with tools looks like this:

```
… <|user|> … <|end|> <|assistant|> [<|analysis|> … <|end|> <|assistant|>] <|tool_call|> {…} <|call|>
  <|tool|> {host result} <|end|> <|assistant|> <|final|> … <|eot|>
```

## 3. Authority and injection resistance

- The authority order is system > developer > user > tool / quoted data. The host assigns
  every role. The model can only produce `<|assistant|>` messages.
- **Text never produces control IDs.** The tokenizer has `encode_special_tokens = True`,
  so a literal `<|system|>` inside user text, a webpage, or a tool result encodes as
  ordinary bytes. Tested: `test_forged_delimiters_in_user_and_tool_data` and
  `test_literal_control_strings_never_become_control_ids`. No textual escaping exists or
  is needed.
- **Integration hazard (measured).** `tokenizer.json` does *not* persist
  `encode_special_tokens` (tokenizers 0.23.2). A consumer that loads it with plain
  `Tokenizer.from_str` turns literal `<|user|>` into control ID 3. Every consumer must
  therefore use `j1data.jtok.load` or set the flag itself. The test
  `test_flag_is_not_persisted_so_load_must_set_it` pins this behaviour.
- **Runtime requirement.** Template-string runtimes such as llama.cpp render a Jinja chat
  template to text and then tokenize it with special-token parsing on. That re-opens
  injection unless the runtime tokenizes role content separately or the App strips or
  escapes control strings before templating. Juniper App must use a token-level
  serializer, or must reject or neutralize the literal control strings in untrusted
  content before rendering. This is a P17 gate item.
- **Sampling mask.** During generation the runtime should mask the IDs the model must never
  emit: `<|system|>`, `<|developer|>`, `<|user|>`, `<|tool|>`, `<|endoftext|>`, and
  reserved IDs 12–63. The parser rejects them anyway.

## 4. Malformed output handling

`parse_generation` never raises. It returns `stop ∈ {call, eot, malformed}`, the complete
messages before the fault, and a reason. The tested cases are:

- a missing channel
- an unterminated message
- the wrong terminator
- a forbidden or reserved control token inside content
- no `<|assistant|>` between messages
- a non-JSON or wrong-shape tool call
- trailing tokens after `<|call|>` or `<|eot|>`

On `malformed` the App neither executes nor displays a partial tool request. It may show
the partial final text marked as incomplete.

## 5. Reconciliation with Juniper App (`juniper-tool-protocol-v1`)

| App concept (`src/types/index.ts`, `src-tauri/src/tools.rs`) | juniper-chat-v1 |
|---|---|
| `NormalizedToolCall {id, name, arguments}` | `CALL_JSON {name, arguments}`; the **host** assigns `id` |
| `HostToolResult {protocolVersion, callId, name, status, result, error}` | the exact JSON body of a `<|tool|>` message |
| `validate_call` (allow-list, argument schemas, 64 KiB payload) | unchanged; the host validates after `<|call|>` |
| `MAX_TOOL_ROUNDS = 4`, `MAX_TOOL_CALLS_PER_ROUND = 8` | one call per `<|call|>`; multiple calls in a round are sequential call/result pairs; the host still enforces both limits |
| Raw reasoning never sent to the UI | `<|analysis|>` content is withheld from display by default (App policy) |
| OpenAI-compatible `tool_calls` from the runtime | the runtime adapter maps `<|tool_call|>…<|call|>` to `tool_calls` and `<|tool|>` back to tool messages |

The App already treats model output as a request and authors results itself. That
security boundary must remain in the host. This table is a contract, not an
implemented Juniper 1 adapter: the App still needs a token-safe serializer,
generation parser, and runtime mapping, followed by end-to-end tests against
the actual text-in/text-out backend. Until those tests pass, native Juniper 1
tool integration is unqualified.

## 6. Loss masking (SFT)

The trainable tokens are those the model produces: the channel token, the content and the
terminator of assistant messages in the selected turns, plus `<|assistant|>` headers the
model emits between its own messages. Host-supplied headers, system, developer, user and
tool messages are never trained. Reasoning (analysis) training is a per-dataset switch
(`train_analysis`). Tested: `test_serialize_masks_and_drops_old_analysis` and
`test_no_analysis_training`.

Pretraining packs use only `<|endoftext|>` between documents. Protocol tokens appear in
training only from post-training data.

## 7. Context budgeting (`fit_to_budget`)

1. The window is the qualified context (2,048, or 4,096 once qualified) minus a reserve
   for generation.
2. Tool results longer than `max_tool_tokens` are replaced by a host-marked truncated
   result (`{"truncated": true, …}`). The result keeps its `<|tool|>` role.
3. Analysis from completed turns is dropped.
4. Whole completed turns are dropped, oldest first. The system and developer messages and
   the current turn are never dropped. If they alone exceed the window, the call raises
   rather than silently cutting authority content.
5. Roles are never merged, and no data is promoted to a higher role.

Default budget proposal at 2,048 tokens:

| Content | Budget |
|---|---:|
| System + developer + tool schemas | 512 |
| Generation reserve | 384 |
| Analysis cap within a turn | 256 |
| Per tool result | 384 |
| Remainder: conversation | ~512 |

These proposals are frozen with the post-training data design, not here.
