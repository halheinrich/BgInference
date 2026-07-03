namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgGame_Lib;
using BgInference;

/// <summary>
/// The v1 threshold cube policy: boundary semantics on both constants, and —
/// load-bearing — that the response side reads the responder's own frame
/// directly. Under the unified perspective convention the responder is queried
/// in its own frame, so no negation is applied; the two-direction pin guards
/// against a stray one being reintroduced.
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
    [InlineData(-0.5f, CubeAction.Take)]     // own equity −0.5: dead-cube boundary, inclusive
    [InlineData(-0.51f, CubeAction.Pass)]    // own equity −0.51: below the take point
    [InlineData(0f, CubeAction.Take)]
    [InlineData(0.3f, CubeAction.Take)]      // a bad double: happily taken
    public async Task Response_TakesAtOrAboveTakePoint(float ownEquity, CubeAction expected)
    {
        // Responder frame: the evaluator's folded equity is the responder's own.
        var response = await AgentSeeing(ownEquity).ChooseResponseAsync(SomeState());
        Assert.Equal(expected, response);
    }

    [Fact]
    public async Task Response_ReadsResponderFrameEquity_BothDirections()
    {
        // The state reaching the responder is already in the responder's own
        // frame, so the folded equity is read directly.
        //
        // Crushed responder (own equity −2): −2 < −0.5 correctly Passes; a stray
        // negation would flip this to +2 ≥ −0.5 and wrongly Take.
        Assert.Equal(CubeAction.Pass, await AgentSeeing(-2f).ChooseResponseAsync(SomeState()));

        // Dominant responder (own equity +2): +2 ≥ −0.5 correctly Takes; a stray
        // negation would flip this to −2 < −0.5 and wrongly Pass.
        Assert.Equal(CubeAction.Take, await AgentSeeing(2f).ChooseResponseAsync(SomeState()));
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
