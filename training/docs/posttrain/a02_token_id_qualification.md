# A-02 remediation: Juniper 1 runs behind host-side j1tok and token IDs only

Finding A-02 (audit 2026-10-12): llama.cpp's own text tokenizer for the exported GGUF reproduced
j1tok on 0/200 validation documents, and with special parsing it turned literal `<|system|>` /
`<|eot|>` text into control IDs. Decision: **Option A, host-side canonical tokenization**. Option B
(a native Juniper pre-tokenizer inside llama.cpp) was rejected: parity would rest on tests rather than
construction, it would fork the pinned runtime, and llama-server's text endpoints (`/tokenize`,
string `/completion`, chat templates) would still exist. Research record with code citations:
FLOWBOX `~/j1post/research/app/a02_runtime_findings.md` (summarized below).

## What was built

| component | where | role |
|---|---|---|
| `TokenIdClient` | Juniper-1-Data `src/j1data/runtime_client.py` | host serializes with `jtok.load` + `protocol.serialize`; `POST /completion` with an all-integer prompt; integer-keyed `logit_bias` bans `{0,1,2,3,5,12..63}`; non-streaming `return_tokens`; resubmits after `<|end|>` closes an analysis message; returns only turns `parse_generation` accepts; literal loopback `http` only, redirects refused, text endpoints refused by construction; startup self-test |
| GGUF markers | `tools/export_gguf.py` | `juniper.io_contract = "token-ids-only"`, `juniper.protocol`, `juniper.tokenizer_sha256`, source manifest digest; the exporter now verifies the source checkpoint and its state digest |
| App guard | Juniper-App `src-tauri/src/gguf_guard.rs` (branch `a02/juniper1-text-path-guard`) | refuses any GGUF with a `juniper.*` key or a `general.name` starting `juniper-1` before the Ollama import, the desktop llama-server launch and the Android JNI launch; malformed headers fail closed |
| live qualification | `qual/a02_token_id_qual.py` | drives llama-server b11270 (748d4225b) only through the client |

Verified llama-server facts (source + probes, both e107984bc and 748d4225b): an all-integer prompt is
used verbatim (no BOS); generated IDs come back only in the non-streaming result (streaming drops IDs
of incomplete UTF-8 byte tokens; e107984bc also returns HTTP 500 on invalid UTF-8, so Juniper 1 is
pinned to 748d4225b); IDs 0, 9, 10, 11 are end-of-generation (so `<|end|>` stops after analysis and
the host resubmits); `logit_bias` works by integer ID; the text endpoints cannot be disabled, so the
host must never call them for Juniper 1.

## Live results (llama-server b11270 / 748d4225b, CPU, F16 exports, client calls `/completion` only)

Frozen base GGUF `41e21fed…b8980` (`results/a02_token_id_qual_base.json`) and the qualified SFT pilot
candidate `ckpt-000001226-55d13f9a` GGUF `1537fde2…95175` (`results/a02_token_id_qual_candidate.json`).

| check | base | SFT candidate |
|---|---|---|
| self-test: 9/10/11 end generation; banned `<|system|>` never sampled | pass | pass |
| 200-document regression set evaluated as exact canonical IDs | 200/200 | 200/200 |
| same documents through llama.cpp's text tokenizer (unsafe path, contrast) | 0/200 identical | 0/200 identical |
| all 64 control spellings in user text: forged control IDs in the prompt | 0 | 0 |
| turns on those 64 prompts (24-token budget) | {'malformed': 64} | {'eot': 6, 'malformed': 58} |
| Python/whitespace/Unicode/U+2028/NUL round trips; serialization prefix stability | pass | pass |
| multi-turn conversation outcome | malformed | eot |
| forced byte-fallback IDs returned intact (non-streaming) | pass | pass |
| malformed generations reported as `malformed` | 2/2 | 2/2 |
| banned IDs under high-temperature sampling (24 seeds) | 0 of 1152 tokens | 0 of 507 tokens |
| KV-cache continuation (split inside the generated span) equals uninterrupted greedy | 3/3 | 3/3 |
| greedy sequences identical to PyTorch fp32 (48 dev prompts × 24 tokens) | 48/48 | 47/48 |

The base never learned the protocol, so its turns parse as `malformed` (nothing is shown to a user).
The candidate ends turns with a well-formed `<|eot|>`. The one candidate greedy divergence is an F16
near-tie; logit tolerance is gated separately by `qual/gguf_parity.py`. Correction recorded: the first
version of the KV-cache check split after the stop token once the model learned to stop (an invalid
comparison that reported a false failure); both models were rerun with the corrected check.

Candidate turn latency through this path (`results/sft_pilot_v1/inference_latency.json`): median 0.13 s
for short turns on CPU (~113 generated tok/s), 0.04 s on the RTX 2060 (~350 tok/s).

## Not done (App integration remains blocked by design)

The native Rust adapter in Juniper App (tokenizers 0.23.2 with `set_encode_special_tokens`, a
protocol port with golden tests, a `Backend::Juniper1` route, a CPU build of 748d4225b) is not
implemented. Until it exists and passes live end-to-end tests including negative cases, Juniper 1
is not offered in the App and the guard refuses it on every text path. Host authority (roles,
permissions, tool execution, tool-result authorship) stays with the existing Rust host code, which the
future adapter must reuse unchanged.
