namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgGame_Lib;
using BgInference;

/// <summary>
/// The v1 threshold cube policy: boundary semantics on both constants, and —
/// load-bearing — that the response side negates. Both decisions are queried
/// with the on-roll-frame snapshot (halheinrich/backgammon#533), and the
/// evaluator answers for the player on roll, the offerer; the responder is the
/// other side, so its own equity is the negation. The two-direction pins guard
/// against a missing or a doubled negation, and against a mirrored board.
/// </summary>
public sealed class ThresholdCubeAgentTests
{
    private static GameSnapshot SomeState() => GameState.NewGame(MatchState.NewMatch(0)).Snapshot();

    /// <summary>
    /// Asymmetric, so neither a fixed position nor the mirror of this one can
    /// stand in for it.
    /// </summary>
    private static GameSnapshot Asymmetric()
    {
        var counts = new int[26];
        counts[6] = 3;
        counts[8] = 2;
        counts[19] = -4;
        var position = new BoardPosition(counts);
        Assert.NotEqual(position, position.Flipped());
        return GameState.FromPosition(
            MatchState.NewMatch(0), new BoardState(position), cubeSize: 1, CubeOwner.Centered).Snapshot();
    }

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
    [InlineData(0.5f, CubeAction.Take)]      // responder's own equity −0.5: dead-cube boundary, inclusive
    [InlineData(0.51f, CubeAction.Pass)]     // responder's own equity −0.51: below the take point
    [InlineData(0f, CubeAction.Take)]
    [InlineData(-0.3f, CubeAction.Take)]     // a bad double: happily taken
    public async Task Response_TakesAtOrAboveTakePoint(float onRollEquity, CubeAction expected)
    {
        // The evaluator's folded equity is the on-roll offerer's; the
        // responder's own is its negation.
        var response = await AgentSeeing(onRollEquity).ChooseResponseAsync(SomeState());
        Assert.Equal(expected, response);
    }

    [Fact]
    public async Task Response_Take_FromAKnownOnRollEquity_OnTheUnchangedBoard()
    {
        // The offerer stands at −2 (the responder at +2): +2 ≥ −0.5 Takes. A
        // missing negation reads −2 < −0.5 and wrongly Passes.
        var state = Asymmetric();
        var evaluator = new TestEvaluators.Recording(new PositionEvaluation(-2f, 0f, 0f, 0f, 0f, 0f));

        var response = await new ThresholdCubeAgent(evaluator).ChooseResponseAsync(state);

        Assert.Equal(CubeAction.Take, response);
        Assert.Equal([state.Board], evaluator.Seen);   // the on-roll frame, not a mirror
    }

    [Fact]
    public async Task Response_Pass_FromAKnownOnRollEquity_OnTheUnchangedBoard()
    {
        // The offerer stands at +2 (the responder at −2): −2 < −0.5 Passes. A
        // missing negation reads +2 ≥ −0.5 and wrongly Takes.
        var state = Asymmetric();
        var evaluator = new TestEvaluators.Recording(new PositionEvaluation(2f, 0f, 0f, 0f, 0f, 0f));

        var response = await new ThresholdCubeAgent(evaluator).ChooseResponseAsync(state);

        Assert.Equal(CubeAction.Pass, response);
        Assert.Equal([state.Board], evaluator.Seen);   // the on-roll frame, not a mirror
    }

    [Fact]
    public async Task BothDecisions_EvaluateTheSnapshotsOwnPosition()
    {
        var state = Asymmetric();
        var evaluator = new TestEvaluators.Recording(new PositionEvaluation(0f, 0f, 0f, 0f, 0f, 0f));
        var agent = new ThresholdCubeAgent(evaluator);

        await agent.ChooseOfferAsync(state);
        await agent.ChooseResponseAsync(state);

        Assert.Equal([state.Board, state.Board], evaluator.Seen);
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
