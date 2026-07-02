namespace BgInference;

using System.Diagnostics;
using BgDataTypes_Lib;

/// <summary>
/// Encodes a board position into the 303-element float32 feature vector consumed
/// by BgRLEngine-exported ONNX models.
///
/// <para>
/// This is a line-for-line mirror of the producer's <c>encode_board</c>
/// (BgRLEngine, <c>engine/state.py</c>). Layout, in order: player points 24×6,
/// opponent points 24×6 (each point a depth-5 thermometer plus one overflow
/// unit), player bar 3, opponent bar 3 (thermometer), player borne-off 2,
/// opponent borne-off 2 (fraction + all-off flag), then 5 global features
/// (player-to-move, pip ratio, race flag, normalized player and opponent
/// checker counts). The executable contract is the committed parity fixture
/// (<c>BgRLEngine/parity/vectors.json</c>), pinned bit-exact at
/// <c>feature_tolerance_abs</c> 0.0. Any producer-side encoding change bumps
/// <c>bgrl.encoding_version</c> and regenerates the fixture in the same commit.
/// </para>
///
/// <para>
/// Float discipline: every stored value is computed in <see langword="double"/>
/// and then narrowed to <see langword="float"/>, matching numpy's
/// compute-in-float64, store-in-float32 behavior. Do not reorder arithmetic.
/// </para>
/// </summary>
internal static class FeatureEncoder
{
    /// <summary>
    /// The board→feature encoding contract version this encoder implements.
    /// Exported models advertise theirs via <c>bgrl.encoding_version</c>
    /// metadata; a mismatch must fail model loading (the cross-language
    /// fail-fast handshake).
    /// </summary>
    internal const int EncodingVersion = 1;

    /// <summary>Feature vector length — the model's input width.</summary>
    internal const int FeatureSize = 303;

    /// <summary>Checkers per player in a legal position; normalization divisor.</summary>
    internal const int CheckersPerPlayer = 15;

    private const int NumPoints = 24;
    private const int UnitsPerPoint = 6;
    private const int ThermometerDepth = 5;
    private const double OverflowScale = 10.0;
    private const int BarFeatures = 3;
    private const int BorneOffFeatures = 2;

    /// <summary>
    /// Encode a position into <paramref name="destination"/>.
    /// </summary>
    /// <param name="board">
    /// The position, in <see cref="BoardState"/>'s on-roll-relative frame
    /// (positive = the perspective player's checkers; <c>Points[25]</c> their
    /// bar, <c>Points[0]</c> the opponent's bar, stored ≤ 0). The perspective
    /// player is the "player" of every player/opponent feature pair.
    /// </param>
    /// <param name="playerToMove">
    /// Whether the perspective player is on roll (feature 298). Evaluating a
    /// board in its natural on-roll frame passes <see langword="true"/>; the
    /// producer's training-record path also encodes positions from the
    /// non-mover's perspective, which is why the flag exists and why the
    /// parity fixture covers both values.
    /// </param>
    /// <param name="offPlayer">
    /// Perspective player's borne-off count. Explicit because
    /// <see cref="BoardState"/> does not store offs; public callers derive
    /// <c>15 − (board + bar)</c>, while parity tests supply the fixture's raw
    /// value so degenerate cases are pinned without trusting the derivation.
    /// </param>
    /// <param name="offOpponent">Opponent's borne-off count; same contract as <paramref name="offPlayer"/>.</param>
    /// <param name="destination">Receives the features; length must be exactly <see cref="FeatureSize"/>.</param>
    /// <exception cref="ArgumentException"><paramref name="destination"/> has the wrong length.</exception>
    internal static void Encode(
        BoardState board,
        bool playerToMove,
        int offPlayer,
        int offOpponent,
        Span<float> destination)
    {
        if (destination.Length != FeatureSize)
        {
            throw new ArgumentException(
                $"Destination must have length {FeatureSize}, got {destination.Length}.",
                nameof(destination));
        }

        destination.Clear();
        int idx = 0;

        // Player points (24 × 6 = 144)
        for (int p = 1; p <= NumPoints; p++)
        {
            EncodePoint(Math.Max(0, board.Points[p]), destination.Slice(idx, UnitsPerPoint));
            idx += UnitsPerPoint;
        }

        // Opponent points (24 × 6 = 144)
        for (int p = 1; p <= NumPoints; p++)
        {
            EncodePoint(Math.Max(0, -board.Points[p]), destination.Slice(idx, UnitsPerPoint));
            idx += UnitsPerPoint;
        }

        // Bar (3 + 3)
        int barPlayer = board.Points[25];
        int barOpponent = -board.Points[0];
        EncodeBar(barPlayer, destination.Slice(idx, BarFeatures));
        idx += BarFeatures;
        EncodeBar(barOpponent, destination.Slice(idx, BarFeatures));
        idx += BarFeatures;

        // Borne off (2 + 2)
        EncodeBorneOff(offPlayer, destination.Slice(idx, BorneOffFeatures));
        idx += BorneOffFeatures;
        EncodeBorneOff(offOpponent, destination.Slice(idx, BorneOffFeatures));
        idx += BorneOffFeatures;

        // Global features (5)
        destination[idx++] = playerToMove ? 1f : 0f;

        int playerPips = board.PipCount;
        int opponentPips = board.OpponentPipCount;
        int totalPips = playerPips + opponentPips;
        destination[idx++] = totalPips > 0 ? (float)((double)playerPips / totalPips) : 0.5f;

        // Race flag. BoardState.IsRace matches the producer's is_race() except
        // when a bar is occupied while the other side has no checkers at all:
        // the producer returns false for *any* occupied bar, where IsRace's
        // positional scan calls that vacuously true. The explicit bar guard
        // pins the producer's semantics.
        bool isRace = barPlayer <= 0 && barOpponent <= 0 && board.IsRace;
        destination[idx++] = isRace ? 1f : 0f;

        destination[idx++] = (float)(PlayerCheckerCount(board) / (double)CheckersPerPlayer);
        destination[idx++] = (float)(OpponentCheckerCount(board) / (double)CheckersPerPlayer);

        Debug.Assert(idx == FeatureSize, "Feature layout drifted from FeatureSize.");
    }

    /// <summary>
    /// Thermometer + overflow encoding of one point's checker count: units 0–4
    /// are 1.0 for each checker up to five; unit 5 is
    /// <c>min((count − 5) / 10.0, 1.0)</c> for counts above five.
    /// </summary>
    private static void EncodePoint(int count, Span<float> slot)
    {
        for (int i = 0; i < Math.Min(count, ThermometerDepth); i++)
            slot[i] = 1f;
        if (count > ThermometerDepth)
            slot[ThermometerDepth] = (float)Math.Min((count - ThermometerDepth) / OverflowScale, 1.0);
    }

    /// <summary>Bar count as a depth-3 thermometer (1, 2, 3+ checkers).</summary>
    private static void EncodeBar(int count, Span<float> slot)
    {
        for (int i = 0; i < Math.Min(count, BarFeatures); i++)
            slot[i] = 1f;
    }

    /// <summary>Borne-off count as (fraction of 15, all-off flag).</summary>
    private static void EncodeBorneOff(int count, Span<float> slot)
    {
        slot[0] = (float)(count / (double)CheckersPerPlayer);
        slot[1] = count >= CheckersPerPlayer ? 1f : 0f;
    }

    /// <summary>Perspective player's checkers in play: on the board plus on the bar.</summary>
    private static int PlayerCheckerCount(BoardState board)
    {
        int onBoard = 0;
        for (int p = 1; p <= NumPoints; p++)
            onBoard += Math.Max(0, board.Points[p]);
        return onBoard + board.Points[25];
    }

    /// <summary>Opponent's checkers in play: on the board plus on the bar.</summary>
    private static int OpponentCheckerCount(BoardState board)
    {
        int onBoard = 0;
        for (int p = 1; p <= NumPoints; p++)
            onBoard += Math.Max(0, -board.Points[p]);
        return onBoard - board.Points[0];
    }
}
