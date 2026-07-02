# BgInference — Subproject Instructions

> Collaboration contract: [`../AGENTS.md`](../AGENTS.md)
> Umbrella status & dependency graph: [`../INSTRUCTIONS.md`](../INSTRUCTIONS.md)
> Mission & principles: [`../VISION.md`](../VISION.md)

## Stack

C# / .NET 10 class library + xUnit test project. ONNX Runtime
(`Microsoft.ML.OnnxRuntime`, CPU) for inference. Visual Studio 2026, Windows.

## Solution

`D:\Users\Hal\Documents\Visual Studio 2026\Projects\backgammon\BgInference\BgInference.slnx`

## Repo

https://github.com/halheinrich/BgInference — branch `main`.

## Depends on

- **BgDataTypes_Lib** — `BoardState` (the on-roll-relative board and its
  apply/flip primitives), `Play`/`Move`, `CubeAction`.
- **BgMoveGen** — `MoveGenerator.GeneratePlays` (candidate plays for one-ply
  selection).
- **BgGame_Lib** — `IPlayAgent`/`ICubeAgent` contracts, `GameState`/`MatchState`,
  `MatchRunner` + `SeededDiceSource` (integration proof).
- **BgRLEngine** (cross-language, test-time only) — producer of the ONNX export
  contract and the committed parity fixtures
  (`../BgRLEngine/BgRLEngine/parity/model.onnx` + `vectors.json`), which the
  test project reads in place from the sibling checkout. Not a build
  dependency.

## Directory tree

```
BgInference/
├── BgInference.slnx
├── Directory.Packages.props        CPM — inline Version= is banned
├── INSTRUCTIONS.md
├── BgInference/
│   ├── BgInference.csproj          net10.0; TreatWarningsAsErrors; XML docs enforced
│   ├── FeatureEncoder.cs           internal 303-feature mirror of encode_board
│   ├── IPositionEvaluator.cs       the public evaluation seam
│   ├── OnnxEvaluator.cs            ORT session wrapper: handshake at Load, batch inference
│   ├── PositionEvaluation.cs       six outcome estimates + equity fold
│   ├── EquityWeights.cs            fold weights; Money mirrors compute_equity's default
│   ├── ModelContractException.cs   named fail-fast handshake failures
│   ├── OnePlyPlayAgent.cs          IPlayAgent: one-ply argmax over negated successors
│   └── ThresholdCubeAgent.cs       ICubeAgent v1: crude cubeless-equity thresholds
└── BgInference.Tests/
    ├── BgInference.Tests.csproj
    ├── ParityFixture.cs            locates + parses BgRLEngine's parity fixtures
    ├── ParityGateTests.cs          the five-step cross-language parity gate
    ├── FeatureEncoderTests.cs      per-feature-family unit pins
    ├── OnnxEvaluatorTests.cs       handshake end-to-end + evaluation surface
    ├── PositionEvaluationTests.cs  equity-fold pins
    ├── OnePlyPlayAgentTests.cs     perspective-negation pin + policy contract
    ├── ThresholdCubeAgentTests.cs  threshold boundaries + responder negation
    ├── MatchIntegrationTests.cs    seeded full matches via MatchRunner
    └── TestAgents.cs               transparent stub evaluators + trivial bots
```

## Architecture

**What this library is.** The C# consumer of BgRLEngine's ONNX export: it
loads exported models, reproduces the producer's 303-feature board encoding
bit-exactly, evaluates positions, and packages that as BgGame_Lib agents — the
first real engine that can enter matches through `MatchRunner`.

**The cross-language contract.** The feature encoding cannot be single-sourced
across Python and C#, so the producer commits an executable contract:
`parity/model.onnx` (tiny deterministic 303→16→16→6 net) plus `vectors.json`
(28 golden board→features→output triples, sha256-paired to the model, with
tolerances in the header). The consumer-side **parity gate**
(`ParityGateTests`) asserts, in contract order and fail-loud: (1) fixture
files exist → (2) model sha256 matches the vectors header → (3)
`bgrl.encoding_version` matches `FeatureEncoder.EncodingVersion` →
(4) board→features bit-exact (`feature_tolerance_abs` 0.0) → (5)
features→output within `output_tolerance_abs` (per-case and all cases in one
batched `Run`, pinning the dynamic batch dimension). Graph contract: input
`features` float32 `[batch, 303]`, output `probabilities` float32 `[batch, 6]`,
opset 17; metadata rides in the ONNX `metadata_props` as `bgrl.*` keys.

**FeatureEncoder (internal).** Line-for-line mirror of `encode_board`
(BgRLEngine `engine/state.py`): per-point depth-5 thermometer + overflow unit
×24 ×2 sides, depth-3 bar thermometers, borne-off fraction + all-off flag, and
five globals (player-to-move, pip ratio, race flag, normalized checker
counts). Every stored value is computed in `double` and narrowed to `float`
(numpy's compute-float64/store-float32). Pip counts reuse
`BoardState.PipCount`/`OpponentPipCount`; the race flag reuses
`BoardState.IsRace` behind an explicit bar-occupancy guard (see Pitfalls).
Board mapping: producer `points[i]` = `Points[i+1]`, bars = `Points[25]` /
`−Points[0]`, borne-off counts derived (see Pitfalls), `player_to_move` is an
encoder parameter, not a board property.

**OnnxEvaluator.** `Load` is a fail-fast handshake: not-loadable ONNX,
missing/malformed `bgrl.*` keys, encoding-version mismatch, structural-key
mismatch, or graph-shape drift each throw a named `ModelContractException`
before any evaluation is possible. Evaluation derives off counts, encodes
`playerToMove: true` (the public path is always the on-roll perspective), and
runs one ORT `Run` per batch. Sessions are thread-safe for inference;
`Dispose` releases the native session. `ModelMetadata` exposes the validated
`bgrl.*` map for diagnostics and future routing (`bgrl.model_role`).

**Agents are thin policies over `IPositionEvaluator`.** `OnePlyPlayAgent`
mirrors the producer's `select_play`: copy → `ApplyPlay` (moves **and** flip
into the opponent's frame) → batched evaluation → **negate** the folded equity
→ strict-improvement argmax (first-max tie-break). Single-legal-play turns
(dance/forced) return without consulting the evaluator; terminal successors
are evaluated like any other, same as `select_play`. `ThresholdCubeAgent` is
the deliberately crude v1 cube (see its XML docs for what it ignores and what
replaces it). Scratch strategy is copy-and-`ApplyPlay`, not apply/undo — right
for one ply × ~20 candidates; apply/undo becomes relevant only with deep
search.

**Equity fold.** `EquityWeights.Money` = `(1, 2, 3, −1, −2, −3)`, mirroring
the producer's `compute_equity` default: the six outputs are treated as
exclusive outcome classes. This is deliberately **not** the cumulative-
probability fold `(1, 1, 1, −1, −1, −1)`.

## Public API

```csharp
public interface IPositionEvaluator
{
    // Board is on-roll-relative; the evaluation is from the on-roll player's
    // perspective. Successors produced by ApplyPlay are in the OPPONENT's
    // frame — negate the folded equity to reason from the mover's side.
    PositionEvaluation Evaluate(BoardState board);
    PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardState> boards); // one Run; empty → empty
}

public sealed class OnnxEvaluator : IPositionEvaluator, IDisposable
{
    public static OnnxEvaluator Load(string modelPath);
    // throws FileNotFoundException | ModelContractException (named violation)
    public IReadOnlyDictionary<string, string> ModelMetadata { get; } // validated bgrl.* map
}

public readonly record struct PositionEvaluation(
    float PWin, float PWinGammon, float PWinBackgammon,
    float PLose, float PLoseGammon, float PLoseBackgammon)
{
    public float Equity(in EquityWeights weights);
}

public readonly record struct EquityWeights(
    float Win, float WinGammon, float WinBackgammon,
    float Lose, float LoseGammon, float LoseBackgammon)
{
    public static EquityWeights Money { get; } // (1, 2, 3, −1, −2, −3)
}

public sealed class ModelContractException : Exception;

public sealed class OnePlyPlayAgent : IPlayAgent
{
    public OnePlyPlayAgent(IPositionEvaluator evaluator);                       // Money fold
    public OnePlyPlayAgent(IPositionEvaluator evaluator, EquityWeights weights);
}

public sealed class ThresholdCubeAgent : ICubeAgent
{
    public const float DoubleEquityThreshold; // +0.4  — offer at or above
    public const float TakeEquityThreshold;   // −0.5  — take at or above (own equity)
    public ThresholdCubeAgent(IPositionEvaluator evaluator);
}
```

Loading a real (non-parity) model: export from BgRLEngine via
`export_onnx.py`, then `OnnxEvaluator.Load(path)` — the handshake accepts any
export whose encoding version matches; nothing here is parity-model-specific.

## Pitfalls

- **Three perspective boundaries live in this library; each has a named
  pinning test.** (1) Successor evaluation: `ApplyPlay` flips the board, so
  `OnePlyPlayAgent` negates folded equity —
  `NegationPin_ChoosesTheHit_WhereUnNegatedArgmaxProvablyWouldNot`.
  (2) Cube response: `MatchRunner` hands the responder a state still in the
  *offerer's* frame, so `ThresholdCubeAgent.ChooseResponseAsync` negates —
  `Response_NegatesTheOffererFrame_BothDirections`. (3) The encoder's
  `player_to_move` flag: the public evaluator always encodes `true`; the flag
  exists for the producer's training-record encodings and is exercised by the
  fixture's `player_to_move: false` cases — parity gate +
  `PlayerToMove_FlipsExactlyOneFeature`. A sign change that dodges its pin is
  a bug in the pin, not a green light.
- **Bit-exactness is arithmetic-shape-sensitive.** Compute in `double`, narrow
  to `float`, per slot. "Simplifying" `(float)(count / 15.0)` to float
  division, or reordering the pip-ratio arithmetic, can break the encoding pin
  without being wrong-looking. The gate reads its tolerances from the fixture
  header — never loosen them locally to get green.
- **`BoardState.IsRace` is not the producer's `is_race()`.** The producer
  returns false whenever *either* bar is occupied — even when the other side
  has no checkers and `IsRace`'s positional scan says vacuously true. The
  encoder's explicit bar guard pins the producer's semantics
  (`Race_PlayerOnBar_IsZero_EvenWhenIsRaceIsVacuouslyTrue`).
- **Borne-off counts are derived, and only on the public path.**
  `BoardState` stores board + bar checkers only; `OnnxEvaluator` derives
  `15 − in-play` per side and fails loud if negative (>15 checkers a side).
  The parity tests deliberately bypass the derivation and feed the fixture's
  raw off counts through the internal encoder, so degenerate fixture cases pin
  the encoding without trusting the derivation.
- **Encoding-version changes are a paired dance.** The producer bumps
  `bgrl.encoding_version` and regenerates both fixtures in one commit; this
  library then updates `FeatureEncoder.EncodingVersion`, adapts the encoder,
  and must re-pass the whole gate. A version mismatch fails at `Load` by
  design — do not catch-and-continue around `ModelContractException`.
- **Never copy the parity fixtures into this repo.** They are consumed in
  place from the BgRLEngine sibling checkout (single source of truth); the
  gate's sha256 step exists precisely so a stale pairing can't pass.
- **The equity fold is the exclusive-classes fold.** `(1, 2, 3, −1, −2, −3)`
  per `compute_equity` — not the GNU-style cumulative fold. The
  `MoneyWeights_MirrorComputeEquityDefault` pin carries the rationale; don't
  "fix" it to `(1, 1, 1, …)`.

## Subproject-internal next steps

- **Terminal-win short-circuit in `OnePlyPlayAgent`** — v1 mirrors
  `select_play` (terminal successors are evaluated like any other); a
  certain-win shortcut is strictly better play and deserves its own pin when
  added.
- **MET/Janowski cube agent** — the recorded replacement for
  `ThresholdCubeAgent`; `EquityWeights` already parameterizes the fold it
  will need.
- **Deeper search (n-ply / rollouts)** — if it comes, that is when the
  apply/undo hot-path pattern (and an encoder path that avoids successor
  copies) becomes worth adopting.
