namespace BgInference;

/// <summary>
/// A model's six outcome estimates for one position, from the evaluated
/// player's perspective. The slot semantics are the model contract's
/// <c>bgrl.output_semantics</c>: <c>p_win, p_win_gammon, p_win_backgammon,
/// p_lose, p_lose_gammon, p_lose_backgammon</c>, where the six slots are the
/// exclusive outcome classes the producer's equity fold
/// (<c>engine/network.py::compute_equity</c>) weights independently.
/// </summary>
/// <param name="PWin">Estimate for winning a single game.</param>
/// <param name="PWinGammon">Estimate for winning a gammon.</param>
/// <param name="PWinBackgammon">Estimate for winning a backgammon.</param>
/// <param name="PLose">Estimate for losing a single game.</param>
/// <param name="PLoseGammon">Estimate for losing a gammon.</param>
/// <param name="PLoseBackgammon">Estimate for losing a backgammon.</param>
public readonly record struct PositionEvaluation(
    float PWin,
    float PWinGammon,
    float PWinBackgammon,
    float PLose,
    float PLoseGammon,
    float PLoseBackgammon)
{
    /// <summary>
    /// Fold the six outcome estimates into a single expected value — the dot
    /// product with <paramref name="weights"/>, mirroring the producer's
    /// <c>compute_equity</c>.
    /// </summary>
    /// <param name="weights">How each outcome class contributes; see <see cref="EquityWeights.Money"/>.</param>
    public float Equity(in EquityWeights weights) =>
        (PWin * weights.Win) +
        (PWinGammon * weights.WinGammon) +
        (PWinBackgammon * weights.WinBackgammon) +
        (PLose * weights.Lose) +
        (PLoseGammon * weights.LoseGammon) +
        (PLoseBackgammon * weights.LoseBackgammon);
}
