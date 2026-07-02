namespace BgInference;

/// <summary>
/// Per-outcome-class weights for folding a <see cref="PositionEvaluation"/>
/// into an equity, mirroring the producer's
/// <c>engine/network.py::compute_equity</c>. Match-score-dependent weight
/// sets (DMP, leader/trailer cube states) belong to a future MET arc; v1
/// ships only the cubeless money fold.
/// </summary>
/// <param name="Win">Weight for a single win.</param>
/// <param name="WinGammon">Weight for a gammon win.</param>
/// <param name="WinBackgammon">Weight for a backgammon win.</param>
/// <param name="Lose">Weight for a single loss.</param>
/// <param name="LoseGammon">Weight for a gammon loss.</param>
/// <param name="LoseBackgammon">Weight for a backgammon loss.</param>
public readonly record struct EquityWeights(
    float Win,
    float WinGammon,
    float WinBackgammon,
    float Lose,
    float LoseGammon,
    float LoseBackgammon)
{
    /// <summary>
    /// Cubeless money-play weights <c>(1, 2, 3, −1, −2, −3)</c> — the
    /// producer's <c>compute_equity</c> default, and the fold its self-play
    /// selection trains under.
    /// </summary>
    public static EquityWeights Money => new(1f, 2f, 3f, -1f, -2f, -3f);
}
