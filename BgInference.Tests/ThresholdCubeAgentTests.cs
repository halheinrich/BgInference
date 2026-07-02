namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgGame_Lib;
using BgInference;

/// <summary>
/// The v1 threshold cube policy: boundary semantics on both constants, and —
/// load-bearing — the responder-side perspective negation, since MatchRunner
/// hands the responder a state still in the offerer's on-roll frame.
/// </summary>
public sealed class ThresholdCubeAgentTests
{
    private static GameState SomeState() => GameState.NewGame(MatchState.NewMatch(0));

    /// <summary>An evaluator whose folded money equity for any board is exactly <paramref name="equity"/>.</summary>
    private static ThresholdCubeAgent AgentSeeing(float equity) =>
        new(new TestEvaluators.Constant(new PositionEvaluation(equity, 0f, 0f, 0f, 0f, 0f)));

    [Theory]
    [InlineData(0.8f, CubeAction.Double)]
    [InlineData(0.4f, CubeAction.Double)]    // boundary: threshold is inclusive
    [InlineData(0.39f, CubeAction.NoDouble)]
    [InlineData(0f, CubeAction.NoDouble)]
    [InlineData(-0.6f, CubeAction.NoDouble)]
    public async Task Offer_DoublesAtOrAboveThreshold(float ownEquity, CubeAction expected)
    {
        var offer = await AgentSeeing(ownEquity).ChooseOfferAsync(SomeState());
        Assert.Equal(expected, offer);
    }

    [Theory]
    [InlineData(0.5f, CubeAction.Take)]      // own equity −0.5: dead-cube boundary, inclusive
    [InlineData(0.51f, CubeAction.Pass)]     // own equity −0.51: below the take point
    [InlineData(0f, CubeAction.Take)]
    [InlineData(-0.3f, CubeAction.Take)]     // a bad double: happily taken
    public async Task Response_TakesAtOrAboveTakePoint_InOwnNegatedEquity(
        float offererEquity, CubeAction expected)
    {
        var response = await AgentSeeing(offererEquity).ChooseResponseAsync(SomeState());
        Assert.Equal(expected, response);
    }

    [Fact]
    public async Task Response_NegatesTheOffererFrame_BothDirections()
    {
        // The state reaching the responder is in the offerer's perspective.
        // Crushed responder (offerer equity +2): un-negated, +2 ≥ −0.5 would
        // wrongly Take; negated, −2 correctly Passes.
        Assert.Equal(CubeAction.Pass, await AgentSeeing(2f).ChooseResponseAsync(SomeState()));

        // Dominant responder (offerer equity −2): un-negated, −2 < −0.5 would
        // wrongly Pass; negated, +2 correctly Takes.
        Assert.Equal(CubeAction.Take, await AgentSeeing(-2f).ChooseResponseAsync(SomeState()));
    }

    [Fact]
    public async Task NullState_Throws()
    {
        var agent = AgentSeeing(0f);
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await agent.ChooseOfferAsync(null!));
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await agent.ChooseResponseAsync(null!));
    }

    [Fact]
    public void NullEvaluator_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new ThresholdCubeAgent(null!));
    }

    [Fact]
    public async Task CanceledToken_Throws()
    {
        var agent = new ThresholdCubeAgent(new TestEvaluators.MustNotBeCalled());
        var canceled = new CancellationToken(canceled: true);

        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await agent.ChooseOfferAsync(SomeState(), canceled));
        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await agent.ChooseResponseAsync(SomeState(), canceled));
    }
}
