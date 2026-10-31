# BgInference

> Collaboration contract: [`../AGENTS.md`](../AGENTS.md)
> Umbrella status & dependency graph: [`../INSTRUCTIONS.md`](../INSTRUCTIONS.md)
> Mission & principles: [`../VISION.md`](../VISION.md)
> Open work: the [tracker](https://github.com/halheinrich/backgammon/issues), label `BgInference`.

## What it is for

The C# consumer of BgRLEngine's ONNX export: it loads exported models,
reproduces the producer's 303-feature board encoding, evaluates positions,
and packages that as BgGame_Lib agents that can enter matches through
`MatchRunner`. It is a C# / .NET 10 class library over ONNX Runtime
(`Microsoft.ML.OnnxRuntime`, CPU).

One project outside this repo references it: `BgTournament.Tests`, whose
dogfooding smoke runs these agents over the committed parity model. No
production project references it today.

## Build and test

From this directory:

```bash
dotnet build BgInference.slnx
```

```bash
dotnet test BgInference.slnx
```

Both need the umbrella checkout: the solution takes BgDataTypes_Lib,
BgGame_Lib and BgMoveGen from their sibling submodules.

The suite also needs the BgRLEngine sibling, though the build does not. The
test project reads BgRLEngine's committed parity fixtures in place, from
`../BgRLEngine/BgRLEngine/parity/` (`model.onnx`, `vectors.json`), reached by
`ParityFixture.ParityDir` five levels up from the test output. Without that
checkout `Gate1_FixtureFilesExist` fails, naming the missing file; no test is
skipped for it. That is the arrangement as it stands; where the fixtures are
stored is the open question of halheinrich/backgammon#455. Nothing here reads
the umbrella's `TestData/`.

## Depends on

- **BgDataTypes_Lib** — `BoardPosition` (the position value the evaluator
  takes, each side's borne-off count, and `CheckersPerSide`), `BoardState`
  (the board a `GameState` holds, read through `ToPosition()`, and the pip
  and race rules the encoder reads), `Play`, `CubeAction`.
- **BgMoveGen** — `MoveGenerator.GenerateSuccessors` (each legal play with
  the position it leaves the next mover, for one-ply selection). The tests
  also use `GeneratePlays` and `IsLegalPlay`.
- **BgGame_Lib** — the `IPlayAgent`/`ICubeAgent` contracts and `GameState`.
  The tests also use `MatchState`, and `MatchRunner` + `SeededDiceSource` for
  the integration proof.
- **BgRLEngine** (cross-language, test-time only) — producer of the ONNX
  export contract and the parity fixtures. Not a build dependency.

## Layout

Five projects under `BgInference.slnx`: this repo's two, below, and the three
outside ones it builds from their sibling checkouts (BgDataTypes_Lib,
BgGame_Lib, BgMoveGen). Repo-root `Directory.Build.props` holds the build
policy and `Directory.Packages.props` the package versions.

**`BgInference/`** — the library. Three areas:

- **Evaluation** — `IPositionEvaluator`, the public seam (position in,
  outcome estimates out, single or batched); `OnnxEvaluator`, its ONNX Runtime
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
(`ParityGateTests`), fed by `ParityFixture`; per feature-family encoder pins;
the evaluator's handshake and surface; the equity fold; each agent's
contract; and seeded full matches through `MatchRunner`. `TestAgents` holds
the transparent stub evaluators and the trivial opponent bots.

## Architecture

**The cross-language contract.** The feature encoding cannot be single-sourced
across Python and C#, so the producer commits an executable contract: a tiny
deterministic parity model and golden board→features→output vectors,
sha256-paired, with tolerances in the vectors header. BgRLEngine owns the
contract's definition, the graph and the `bgrl.*` metadata alike
(`engine/export.py` and its `INSTRUCTIONS.md`). This repo checks it with the
**parity gate**, `ParityGateTests`, whose test names carry the step of the
contract, not a run order:

1. `Gate1_FixtureFilesExist` — both fixture files exist.
2. `Gate2_ModelSha256_MatchesVectorsHeader` — the model file's hash equals
   the header's value.
3. `Gate3_MetadataHandshake_ParityModelLoads` — the parity model passes
   `Load`'s handshake, so its `bgrl.encoding_version` matches
   `FeatureEncoder.EncodingVersion`.
4. `Gate4_VectorsHeader_MatchesEncoderContract` — the header states this
   encoder's version and size and a `feature_tolerance_abs` of 0.0; then
   `Gate4_EncodingPin_BitExact` — board→features for each golden case, within
   the tolerance it reads from that header, so bit-exact.
5. `Gate5_InferencePin_FeaturesToOutput` — features→output within
   `output_tolerance_abs`, per case; then
   `Gate5_InferencePin_HoldsForTheWholeBatchInOneRun` — all cases in one
   batched `Run`, pinning the dynamic batch dimension.

Beside the steps, `PublicPipeline_BoardToOutput_MatchesFixtureForOnRollCases`
ties the whole public path to the fixture's on-roll cases.

**FeatureEncoder (internal).** Line-for-line mirror of `encode_board`
(BgRLEngine `engine/state.py`): per-point depth-5 thermometer + overflow unit
×24 ×2 sides, depth-3 bar thermometers, borne-off fraction + all-off flag, and
five globals (player-to-move, pip ratio, race flag, normalized checker
counts). Every stored value is computed in `double` and narrowed to `float`
(numpy's compute-float64/store-float32). Pip counts reuse
`BoardState.PipCount`/`OpponentPipCount`; the race flag reuses
`BoardState.IsRace` behind an explicit bar-occupancy guard (see Pitfalls).
Both are `BoardState`'s rules and `BoardPosition` publishes neither, so the
encoder reads them from a board built on the position: the producer gap of
halheinrich/backgammon#300.
Each side's checker count is the position's, read back from
`BoardPosition.OnRollBorneOffCount`/`OpponentBorneOffCount`. The divisor and
the all-off threshold are the producer's `CHECKERS_PER_PLAYER`, the number of
checkers a side has, so the encoder reads `BoardPosition.CheckersPerSide`,
that number's one statement, rather than naming its own; the overflow scale,
a choice of the encoding, stays the encoder's constant.
Board mapping, into `BoardPosition`'s layout: producer `points[i]` = slot
`i+1`, bars = slot 25 / −slot 0; the borne-off counts and `player_to_move`
are encoder parameters (see Pitfalls).

**OnnxEvaluator.** `Load` is a fail-fast handshake: not-loadable ONNX,
missing/malformed `bgrl.*` keys, encoding-version mismatch, structural-key
mismatch, or graph-shape drift each throw a named `ModelContractException`
before any evaluation is possible; a missing file throws
`FileNotFoundException`. Evaluation takes each side's borne-off count from the
position (`BoardPosition.OnRollBorneOffCount` / `OpponentBorneOffCount`),
encodes `playerToMove: true` (the public path is always the on-roll
perspective), and runs one ORT `Run` per batch. `IPositionEvaluator` requires
safety under concurrent calls; here that rests on ONNX Runtime's
[architecture reference](https://onnxruntime.ai/docs/reference/high-level-design.html)
("Multiple threads can invoke the `Run()` method on the same inference session
object"), and no test in this repo exercises concurrency. `Dispose` releases
the native session. `ModelMetadata` exposes the validated `bgrl.*` map for
diagnostics and future routing (`bgrl.model_role`), as a read-only view over a
private copy that a cast cannot write
(`ModelMetadata_CannotBeWrittenThroughACast`).

To load a real (non-parity) model, export it from BgRLEngine with
`export_onnx.py`, then call `OnnxEvaluator.Load(path)`. The handshake accepts
any export that meets the contract; nothing here is parity-model-specific.

**Positions at the boundary.** `IPositionEvaluator` and the encoder take a
`BoardPosition`, not a `BoardState`: they only read the position. It is the
precedent Hal ruled for `MoveEntryState`'s constructor (2026-09-26, on
halheinrich/backgammon#273): requiring a board makes a caller holding a
position build a richer, mutable object for nothing. The one-ply agent hands
over its successors as positions; the cube agent passes
`GameState.Board.ToPosition()`. A value cannot be null or be
changed by the evaluator, so neither needs saying.

**Agents are thin policies over `IPositionEvaluator`.** `OnePlyPlayAgent`
mirrors the producer's `select_play`: the move generator's successors
(`MoveGenerator.GenerateSuccessors`, each play with the position it leaves
the opponent, in the opponent's frame) → batched evaluation of the positions
→ **negate** the folded equity → strict-improvement argmax (first-max
tie-break, in the generator's candidate order) → the play of the winning
successor. The play and the position it was scored on come from the same
successor, so they cannot be paired wrongly; the agent builds no successor of
its own and copies no board. It scores the evaluations it gets back by index,
over their own length, and does not check their count against the successors
(halheinrich/backgammon#301). Single-legal-play turns (dance/forced) return
without consulting the evaluator; terminal successors are evaluated like any
other, same as `select_play` (a certain-win short-circuit is
halheinrich/backgammon#482). `ThresholdCubeAgent` is the deliberately crude
v1 cube: its XML docs say what it ignores, and a MET/Janowski replacement is
halheinrich/backgammon#483.

**Equity fold.** `EquityWeights.Money` = `(1, 2, 3, −1, −2, −3)`, mirroring
the producer's `compute_equity` default: the six outputs are treated as
exclusive outcome classes, not cumulative probabilities. `EquityWeights`
already parameterizes the fold, which is what a score-dependent cube agent
would need.

## Pitfalls

- **Two perspective boundaries live in this library; each has a named
  pinning test.** (1) Successor evaluation: a successor is in the next
  mover's frame (`MoveGenerator.GenerateSuccessors`), so `OnePlyPlayAgent`
  negates folded equity —
  `NegationPin_ChoosesTheHit_WhereUnNegatedArgmaxProvablyWouldNot`.
  (2) The encoder's `player_to_move` flag: the public evaluator always encodes
  `true`; the flag exists for the producer's training-record encodings and is
  exercised by the fixture's `player_to_move: false` cases — parity gate +
  `PlayerToMove_FlipsExactlyOneFeature`. The cube agent is not a third:
  `ICubeAgent` queries each side in its own frame, so neither decision
  negates, and `Response_ReadsResponderFrameEquity_BothDirections` fails on a
  stray negation. A sign change that dodges its pin is a bug in the pin, not
  a green light.
- **A play travels with its position.** Do not rebuild the pairing by zipping
  `GeneratePlays` with a list of positions, or by flipping and applying plays
  here: the successor rule and the flip are BgMoveGen's and BgDataTypes_Lib's.
  `ChosenPlay_ReachesThePositionScoredBest_ForEverySuccessor` pins it by
  applying the returned play to a copy of the board for every candidate.
- **Bit-exactness is arithmetic-shape-sensitive.** Compute in `double`, narrow
  to `float`, per slot, and do not reorder the arithmetic. Rewriting
  `(float)(count / (double)BoardPosition.CheckersPerSide)` as a multiplication
  by a `float` reciprocal, `count * (1f / BoardPosition.CheckersPerSide)`,
  looks harmless, turns the gate red and differs for six of the sixteen
  counts (measured by mutation, 2026-09-28; the record is the message of
  a5bb026). Plain `float` division is not an example: for one division of
  small integers it rounds as the `double` form does (same record, an
  exhaustive check of the counts and of pip totals to 750), so it passes.
  The gate reads its tolerances from the fixture header — never loosen them
  locally to get green.
- **`BoardState.IsRace` is not the producer's `is_race()`.** The producer
  returns false whenever *either* bar is occupied — even when the other side
  has no checkers and `IsRace`'s positional scan says vacuously true. The
  encoder's explicit bar guard pins the producer's semantics
  (`Race_PlayerOnBar_IsZero_EvenWhenIsRaceIsVacuouslyTrue`).
- **The borne-off counts are encoder inputs; the position's are supplied
  only on the public path.** This library derives no count of its own:
  `OnnxEvaluator` passes the position's (halheinrich/backgammon#295). The
  encoder keeps them as parameters because the producer's board carries its
  off counts independently of its points, and the parity tests feed the
  fixture's raw values through the internal encoder. Do not have the encoder
  read them from the position: the fixture's `bar-one-each` case has 14
  on-roll checkers in play and an off count of 0, so the gate goes red for
  it.
- **Encoding-version changes are a paired dance.** The producer bumps
  `bgrl.encoding_version` and regenerates both fixtures in one commit; this
  library then updates `FeatureEncoder.EncodingVersion`, adapts the encoder,
  and must re-pass the whole gate. A version mismatch fails at `Load` by
  design — do not catch-and-continue around `ModelContractException`.
- **Never copy the parity fixtures into this repo.** Read in place (see Build
  and test), they have one source of truth. The gate's sha256 step compares
  the model file's hash to the value in the vectors header, so a model that
  no longer matches its vectors fails; it does not compare either file with
  the producer's current code.
- **Do not "fix" the equity fold to `(1, 1, 1, …)`,** the GNU-style cumulative
  fold. Equity fold above gives the reason, and
  `MoneyWeights_MirrorComputeEquityDefault` pins it.
