namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgGame_Lib;
using BgInference;
using BgMoveGen;

/// <summary>
/// Stub evaluators with transparent, hand-checkable semantics, plus trivial
/// opponent bots for match integration. The stubs return raw scores in the
/// probability slots — legal because <see cref="PositionEvaluation"/> is a
/// value carrier and the agents only fold and compare.
/// </summary>
internal static class TestEvaluators
{
    /// <summary>
    /// Scores a position as the on-roll player's pip lead:
    /// <c>PWin = OpponentPipCount − PipCount</c> (fewer pips remaining =
    /// leading), other slots zero, so <c>Equity(Money) == pip lead</c>.
    /// Ground truth is computable by hand, which is what the
    /// perspective-negation pin needs.
    /// </summary>
    internal sealed class PipLead : IPositionEvaluator
    {
        public PositionEvaluation Evaluate(BoardPosition position)
        {
            var board = new BoardState(position); // the pip rule is BoardState's
            return new(board.OpponentPipCount - board.PipCount, 0f, 0f, 0f, 0f, 0f);
        }

        public PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardPosition> positions) =>
            positions.Select(Evaluate).ToArray();
    }

    /// <summary>Returns the same evaluation for every position.</summary>
    internal sealed class Constant(PositionEvaluation value) : IPositionEvaluator
    {
        public PositionEvaluation Evaluate(BoardPosition position) => value;

        public PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardPosition> positions) =>
            Enumerable.Repeat(value, positions.Count).ToArray();
    }

    /// <summary>
    /// Scores each position by a caller-supplied rule — for scenarios that
    /// single out one position by value.
    /// </summary>
    internal sealed class Scripted(Func<BoardPosition, PositionEvaluation> evaluate) : IPositionEvaluator
    {
        public PositionEvaluation Evaluate(BoardPosition position) => evaluate(position);

        public PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardPosition> positions) =>
            positions.Select(Evaluate).ToArray();
    }

    /// <summary>
    /// Records every position it is asked about, in order, and returns
    /// <paramref name="value"/> for each — for pinning <em>which</em>
    /// positions a caller evaluates.
    /// </summary>
    internal sealed class Recording(PositionEvaluation value) : IPositionEvaluator
    {
        private readonly List<BoardPosition> _seen = [];

        public IReadOnlyList<BoardPosition> Seen => _seen.ToArray();

        public PositionEvaluation Evaluate(BoardPosition position)
        {
            _seen.Add(position);
            return value;
        }

        public PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardPosition> positions) =>
            positions.Select(Evaluate).ToArray();
    }

    /// <summary>
    /// Fails the test if consulted — guards paths that must not evaluate
    /// (e.g. the single-legal-play short-circuit).
    /// </summary>
    internal sealed class MustNotBeCalled : IPositionEvaluator
    {
        public PositionEvaluation Evaluate(BoardPosition position) =>
            throw new InvalidOperationException("The evaluator must not be consulted on this path.");

        public PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardPosition> positions) =>
            throw new InvalidOperationException("The evaluator must not be consulted on this path.");
    }
}

/// <summary>Always picks the first legal play — deterministic, no evaluation.</summary>
internal sealed class FirstPlayAgent : IPlayAgent
{
    public ValueTask<Play> ChoosePlayAsync(
        GameState state, int die1, int die2, CancellationToken cancellationToken = default)
    {
        var plays = MoveGenerator.GeneratePlays(state.Board, die1, die2);
        return ValueTask.FromResult(plays[0]);
    }
}

/// <summary>Never doubles; takes if doubled.</summary>
internal sealed class NeverCubeAgent : ICubeAgent
{
    public ValueTask<CubeAction> ChooseOfferAsync(
        GameState state, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(CubeAction.NoDouble);

    public ValueTask<CubeAction> ChooseResponseAsync(
        GameState state, CancellationToken cancellationToken = default) =>
        ValueTask.FromResult(CubeAction.Take);
}
