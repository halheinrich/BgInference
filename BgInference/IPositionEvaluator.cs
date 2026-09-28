namespace BgInference;

using BgDataTypes_Lib;

/// <summary>
/// Evaluates backgammon positions: position in, outcome estimates out.
///
/// <para>
/// Contract: a position is read in its own frame, the frame of the player on
/// roll (see <see cref="BoardPosition"/>), and the returned
/// <see cref="PositionEvaluation"/> is from that player's perspective.
/// Callers comparing candidate plays evaluate each play's successor
/// (<c>MoveGenerator.GenerateSuccessors</c>), which is in the <em>next
/// mover's</em> frame, and must negate the folded equity to reason from the
/// mover's side (the producer's <c>select_play</c> convention).
/// </para>
///
/// <para>
/// The input is a <see cref="BoardPosition"/> value, not a
/// <see cref="BoardState"/>: an evaluator only reads the position, so a
/// caller holding a position passes it as it is, and a caller holding a
/// board passes <see cref="BoardState.ToPosition"/>.
/// </para>
///
/// <para>
/// Implementations must be safe for concurrent calls and must evaluate every
/// position, terminal ones included (the empty board, a side with every
/// checker off): a caller like a match loop may evaluate positions where no
/// decision remains.
/// </para>
/// </summary>
public interface IPositionEvaluator
{
    /// <summary>Evaluate a single position.</summary>
    /// <param name="position">The position, in the on-roll player's frame.</param>
    /// <returns>Outcome estimates from the on-roll player's perspective.</returns>
    PositionEvaluation Evaluate(BoardPosition position);

    /// <summary>
    /// Evaluate several positions in one pass (one model invocation for the
    /// whole batch, where the implementation supports it).
    /// </summary>
    /// <param name="positions">The positions, each in its on-roll player's frame.</param>
    /// <returns>One evaluation per position, in input order; empty input yields an empty array.</returns>
    PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardPosition> positions);
}
