namespace BgInference;

using BgDataTypes_Lib;
using BgGame_Lib;

/// <summary>
/// v1 cube policy: fixed thresholds over cubeless money equity.
///
/// <para>
/// <b>Deliberately crude.</b> Both decisions fold the current position with
/// <see cref="EquityWeights.Money"/> and compare against constants: offer a
/// double when own equity is at least <see cref="DoubleEquityThreshold"/>;
/// take when own equity is at least <see cref="TakeEquityThreshold"/> (the
/// classic dead-cube take point). What this ignores — cube ownership value,
/// recube vig, too-good-to-double gammon plays, market-losing sequences, and
/// all match-score dependence — is exactly what the recorded MET/Janowski
/// arc replaces this class with. The constants are contract, not tuning:
/// change them only with tests.
/// </para>
///
/// <para>
/// <b>Perspective.</b> <c>MatchRunner</c> passes the responder a
/// <see cref="GameState"/> still in the <em>offerer's</em> on-roll
/// perspective (see its "Perspective note for cube responders"), so
/// <see cref="ChooseResponseAsync"/> negates the folded equity before
/// comparing — the cube-side twin of the play agent's successor negation.
/// </para>
/// </summary>
public sealed class ThresholdCubeAgent : ICubeAgent
{
    /// <summary>
    /// Offer a double when own cubeless money equity is at least this.
    /// Crude proxy for "market loss is threatening" (roughly a 70% game).
    /// </summary>
    public const float DoubleEquityThreshold = 0.4f;

    /// <summary>
    /// Take a double when own cubeless money equity is at least this — the
    /// dead-cube take point (risk 1 to win 1 ⇒ indifferent at −0.5).
    /// </summary>
    public const float TakeEquityThreshold = -0.5f;

    private readonly IPositionEvaluator _evaluator;

    /// <summary>Create a threshold cube agent over <paramref name="evaluator"/>.</summary>
    /// <param name="evaluator">Scores the current position; shared with the play agent in practice.</param>
    public ThresholdCubeAgent(IPositionEvaluator evaluator)
    {
        ArgumentNullException.ThrowIfNull(evaluator);
        _evaluator = evaluator;
    }

    /// <inheritdoc />
    public ValueTask<CubeAction> ChooseOfferAsync(
        GameState state, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        cancellationToken.ThrowIfCancellationRequested();

        float equity = _evaluator.Evaluate(state.Board).Equity(EquityWeights.Money);
        return ValueTask.FromResult(
            equity >= DoubleEquityThreshold ? CubeAction.Double : CubeAction.NoDouble);
    }

    /// <inheritdoc />
    public ValueTask<CubeAction> ChooseResponseAsync(
        GameState state, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(state);
        cancellationToken.ThrowIfCancellationRequested();

        // The state is in the offerer's on-roll frame; own equity is the negation.
        float offererEquity = _evaluator.Evaluate(state.Board).Equity(EquityWeights.Money);
        float ownEquity = -offererEquity;
        return ValueTask.FromResult(
            ownEquity >= TakeEquityThreshold ? CubeAction.Take : CubeAction.Pass);
    }
}
