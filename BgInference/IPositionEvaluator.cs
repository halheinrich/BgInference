namespace BgInference;

using BgDataTypes_Lib;

/// <summary>
/// Evaluates backgammon positions: board in, outcome estimates out.
///
/// <para>
/// Contract: the board is in <see cref="BoardState"/>'s on-roll-relative
/// frame (positive counts = the player on roll), and the returned
/// <see cref="PositionEvaluation"/> is from that on-roll player's
/// perspective. Callers comparing candidate plays evaluate each successor
/// board — which <see cref="BoardState.ApplyPlay"/> flips to the *opponent's*
/// on-roll frame — and must negate the folded equity to reason from the
/// mover's side (the producer's <c>select_play</c> convention).
/// </para>
///
/// <para>
/// Implementations must be safe for concurrent calls and must tolerate
/// degenerate-but-representable boards (empty board, game over, all checkers
/// off): a caller like a match loop may evaluate positions where no decision
/// remains.
/// </para>
/// </summary>
public interface IPositionEvaluator
{
    /// <summary>Evaluate a single position.</summary>
    /// <param name="board">The position, on-roll-relative. Treated as input only — not mutated.</param>
    /// <returns>Outcome estimates from the on-roll player's perspective.</returns>
    PositionEvaluation Evaluate(BoardState board);

    /// <summary>
    /// Evaluate several positions in one pass (one model invocation for the
    /// whole batch, where the implementation supports it).
    /// </summary>
    /// <param name="boards">The positions, each on-roll-relative. Treated as input only — not mutated.</param>
    /// <returns>One evaluation per board, in input order; empty input yields an empty array.</returns>
    PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardState> boards);
}
