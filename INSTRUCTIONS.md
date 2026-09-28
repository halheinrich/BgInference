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

## Layout

Two projects under `BgInference.slnx`, governed by repo-root
`Directory.Build.props` (TFM, nullable, `TreatWarningsAsErrors`, XML doc
generation) and `Directory.Packages.props` (Central Package Management —
no inline `Version=`).

**`BgInference/`** — the library. Three areas:

- **Evaluation** — `IPositionEvaluator`, the public seam (board in, outcome
  estimates out, single or batched); `OnnxEvaluator`, its ONNX Runtime
  implementation, which runs the fail-fast contract handshake at `Load`;
  `ModelContractException`, the handshake's named failures; and the
  internal `FeatureEncoder`, the mirror of the producer's 303-feature
  `encode_board`.
- **Outcomes and equity** — `PositionEvaluation`, the six outcome
  estimates and their equity fold; `EquityWeights`, the fold weights, whose
  `Money` set mirrors the producer's `compute_equity` default.
- **Agents** — `OnePlyPlayAgent`, the `IPlayAgent` taking the best one-ply
  successor under the negated opponent-frame equity; `ThresholdCubeAgent`,
  the v1 `ICubeAgent` of deliberately crude cubeless-money thresholds.

**`BgInference.Tests/`** — xUnit. The cross-language parity gate
(`ParityGateTests`), fed by `ParityFixture`, which reads BgRLEngine's
committed parity model and vectors in place from the sibling checkout; per
feature-family encoder pins; the evaluator's handshake and surface; the
equity fold; each agent's contract; and seeded full matches through
`MatchRunner`. `TestAgents` holds the transparent stub evaluators and the
trivial opponent bots.

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
Each side's checker count is the position's, read back from
`BoardPosition.OnRollBorneOffCount`/`OpponentBorneOffCount`. The divisor and
the all-off threshold are the producer's `CHECKERS_PER_PLAYER`, the number of
checkers a side has, so the encoder reads `BoardPosition.CheckersPerSide`,
that number's one statement, rather than naming its own; the overflow scale,
a choice of the encoding, stays the encoder's constant.
Board mapping: producer `points[i]` = `Points[i+1]`, bars = `Points[25]` /
`−Points[0]`; the borne-off counts and `player_to_move` are encoder
parameters (see Pitfalls).

**OnnxEvaluator.** `Load` is a fail-fast handshake: not-loadable ONNX,
missing/malformed `bgrl.*` keys, encoding-version mismatch, structural-key
mismatch, or graph-shape drift each throw a named `ModelContractException`
before any evaluation is possible. Evaluation takes each side's borne-off
count from the position (`BoardPosition.OnRollBorneOffCount` /
`OpponentBorneOffCount`), encodes `playerToMove: true` (the public path is always the on-roll perspective), and
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

- **Two perspective boundaries live in this library; each has a named
  pinning test.** (1) Successor evaluation: `ApplyPlay` flips the board, so
  `OnePlyPlayAgent` negates folded equity —
  `NegationPin_ChoosesTheHit_WhereUnNegatedArgmaxProvablyWouldNot`.
  (2) The encoder's `player_to_move` flag: the public evaluator always encodes
  `true`; the flag exists for the producer's training-record encodings and is
  exercised by the fixture's `player_to_move: false` cases — parity gate +
  `PlayerToMove_FlipsExactlyOneFeature`. A sign change that dodges its pin is
  a bug in the pin, not a green light.
  **The cube-response boundary was eliminated**, not pinned: it was a third
  boundary while `MatchRunner` handed the responder a state in the *offerer's*
  frame (so `ChooseResponseAsync` negated). The perspective-unification work
  (BgGame_Lib's responder-frame `ChooseResponseAsync` re-contract — the
  queried player always sees its own frame) deleted that trap class; the
  responder now reads its own folded equity directly, pinned by
  `Response_ReadsResponderFrameEquity_BothDirections`. That two-direction pin
  is what guards against a stray negation being reintroduced.
- **Bit-exactness is arithmetic-shape-sensitive.** Compute in `double`, narrow
  to `float`, per slot. "Simplifying"
  `(float)(count / (double)BoardPosition.CheckersPerSide)` to float
  division, or reordering the pip-ratio arithmetic, can break the encoding pin
  without being wrong-looking. The gate reads its tolerances from the fixture
  header — never loosen them locally to get green.
- **`BoardState.IsRace` is not the producer's `is_race()`.** The producer
  returns false whenever *either* bar is occupied — even when the other side
  has no checkers and `IsRace`'s positional scan says vacuously true. The
  encoder's explicit bar guard pins the producer's semantics
  (`Race_PlayerOnBar_IsZero_EvenWhenIsRaceIsVacuouslyTrue`).
- **The borne-off counts are encoder inputs; the position's are supplied
  only on the public path.** This library derives no count of its own:
  `OnnxEvaluator` passes the position's (`BoardPosition.OnRollBorneOffCount`
  / `OpponentBorneOffCount`, halheinrich/backgammon#295). The encoder keeps
  them as parameters because the producer's board carries its off counts
  independently of its points, and the parity tests feed the fixture's raw
  values through the internal encoder. Do not have the encoder read them
  from the position: the fixture's `bar-one-each` case has 14 on-roll
  checkers in play and an off count of 0, so the gate goes red for it.
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
