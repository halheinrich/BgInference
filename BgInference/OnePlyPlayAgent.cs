namespace BgInference;

using BgDataTypes_Lib;
using BgGame_Lib;
using BgMoveGen;

/// <summary>
/// One-ply play policy over an <see cref="IPositionEvaluator"/>: evaluate
/// every legal play's successor position and take the best.
///
/// <para>
/// This mirrors the producer's <c>engine/game.py::select_play</c> exactly.
/// The candidates are the move generator's successors
/// (<see cref="MoveGenerator.GenerateSuccessors"/>): each legal play with the
/// position it leaves the next mover, in the <em>opponent's</em> frame — so
/// the evaluator scores each successor from the opponent's perspective, and
/// the folded equity is <em>negated</em> before comparison: the best play
/// minimizes the opponent's equity. Ties break to the first maximum, in the
/// generator's candidate order, matching <c>argmax</c>, so play choice is
/// deterministic for a deterministic evaluator.
/// </para>
///
/// <para>
/// The chosen play is taken from the same successor as the position it was
/// scored on, so a play cannot be returned for another play's position.
/// </para>
///
/// <para>
/// When only one legal play exists (a dance, or a fully forced play) it is
/// returned without consulting the evaluator. <c>MatchRunner</c> never
/// queries an agent on a dance, but the short-circuit keeps this agent
/// correct for any driver. Terminal successors are evaluated like any other
/// (no certain-win short-circuit), again mirroring <c>select_play</c>; a
/// shortcut is a recorded candidate improvement, not v1 behavior.
/// </para>
/// </summary>
public sealed class OnePlyPlayAgent : IPlayAgent
{
    private readonly IPositionEvaluator _evaluator;
    private readonly EquityWeights _weights;

    /// <summary>
    /// Create a one-ply agent folding evaluations with
    /// <see cref="EquityWeights.Money"/> — the producer's training fold.
    /// </summary>
    /// <param name="evaluator">Scores candidate successor positions.</param>
    public OnePlyPlayAgent(IPositionEvaluator evaluator)
        : this(evaluator, EquityWeights.Money)
    {
    }

    /// <summary>
    /// Create a one-ply agent folding evaluations with explicit weights
    /// (score-dependent weight selection belongs to a future MET arc).
    /// </summary>
    /// <param name="evaluator">Scores candidate successor positions.</param>
    /// <param name="weights">The equity fold applied to each successor evaluation.</param>
    public OnePlyPlayAgent(IPositionEvaluator evaluator, EquityWeights weights)
    {
        ArgumentNullException.ThrowIfNull(evaluator);
        _evaluator = evaluator;
        _weights = weights;
    }

    /// <inheritdoc />
    public ValueTask<Play> ChoosePlayAsync(
        GameState state,
        int die1,
        int die2,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        cancellationToken.ThrowIfCancellationRequested();

        var successors = MoveGenerator.GenerateSuccessors(state.Board, die1, die2);
        if (successors.Count == 1)
            return ValueTask.FromResult(successors[0].Play); // dance or forced — nothing to decide

        var positions = new BoardPosition[successors.Count];
        for (int i = 0; i < successors.Count; i++)
            positions[i] = successors[i].Position;

        var evaluations = _evaluator.EvaluateBatch(positions);

        int bestIndex = 0;
        float bestEquity = float.NegativeInfinity;
        for (int i = 0; i < evaluations.Length; i++)
        {
            // Successors are in the opponent's on-roll frame: negate to get
            // the mover's equity (select_play's convention).
            float equity = -evaluations[i].Equity(_weights);
            if (equity > bestEquity)
            {
                bestEquity = equity;
                bestIndex = i;
            }
        }

        return ValueTask.FromResult(successors[bestIndex].Play);
    }
}
