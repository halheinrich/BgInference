namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgGame_Lib;
using BgInference;
using BgMoveGen;

/// <summary>
/// The one-ply policy's behavioral contract — above all the
/// perspective-negation pin, the arc's second behavioral proof after the
/// parity gate.
/// </summary>
public sealed class OnePlyPlayAgentTests
{
    private static GameState AtPosition(int[] mop) =>
        GameState.FromPosition(
            MatchState.NewMatch(0), BoardState.FromMop(mop), cubeSize: 1, CubeOwner.Centered);

    private static bool Hits(Play play)
    {
        for (int i = 0; i < play.Count; i++)
        {
            if (play[i].ToPt < 0)
                return true;
        }
        return false;
    }

    /// <summary>
    /// The perspective-negation pin.
    ///
    /// <para>
    /// Applying a play flips the successor into the opponent's on-roll frame,
    /// so the evaluator scores it for the opponent; the agent must negate.
    /// This scenario makes a sign error provably wrong: under the pip-lead
    /// stub, every play spends exactly 3 of the mover's pips (dice 2-1, no
    /// bear-off in range), and hitting the lone blot on point 18 additionally
    /// sets the opponent back 18 pips (from 7 pips-to-home to 25 on the bar).
    /// Every hitting play therefore strictly dominates every non-hitting play
    /// for the mover — while for the opponent it's exactly reversed. The
    /// correct agent must hit; the un-negated argmax provably picks a
    /// non-hitting play (asserted below as the naive pick).
    /// </para>
    /// </summary>
    [Fact]
    public async Task NegationPin_ChoosesTheHit_WhereUnNegatedArgmaxProvablyWouldNot()
    {
        var mop = new int[26];
        mop[20] = 1;  // mover
        mop[13] = 1;  // mover
        mop[18] = -1; // opponent's lone blot: hittable 20/18* with the 2
        mop[1] = -2;  // inert opponent anchor
        var state = AtPosition(mop);
        var plays = MoveGenerator.GeneratePlays(state.Board, die1: 2, die2: 1);

        // Scenario validity: both hitting and non-hitting plays must exist,
        // or the pin is vacuous.
        Assert.Contains(plays, Hits);
        Assert.Contains(plays, p => !Hits(p));

        var agent = new OnePlyPlayAgent(new TestEvaluators.PipLead());
        var chosen = await agent.ChoosePlayAsync(state, die1: 2, die2: 1);

        Assert.True(Hits(chosen),
            "The agent failed to hit — successor equities were not negated (or negated twice).");

        // And pin the trap itself: the naive un-negated argmax over the same
        // successors lands on a non-hitting play.
        var naive = plays[ArgmaxUnNegated(state.Board, plays)];
        Assert.False(Hits(naive),
            "Scenario no longer discriminates: the un-negated pick also hits.");
    }

    private static int ArgmaxUnNegated(BoardState board, List<Play> plays)
    {
        var stub = new TestEvaluators.PipLead();
        int bestIndex = 0;
        float best = float.NegativeInfinity;
        for (int i = 0; i < plays.Count; i++)
        {
            var successor = board.Copy();
            successor.ApplyPlay(plays[i]);
            float equity = stub.Evaluate(successor).Equity(EquityWeights.Money); // no negation
            if (equity > best)
            {
                best = equity;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    [Fact]
    public async Task Dance_ReturnsTheEmptyPlay_WithoutConsultingTheEvaluator()
    {
        var mop = new int[26];
        mop[25] = 1; // mover on the bar
        for (int p = 19; p <= 24; p++)
            mop[p] = -2; // opponent owns the whole entry board
        var state = AtPosition(mop);

        var agent = new OnePlyPlayAgent(new TestEvaluators.MustNotBeCalled());
        var chosen = await agent.ChoosePlayAsync(state, die1: 6, die2: 4);

        Assert.Equal(0, chosen.Count);
    }

    [Fact]
    public async Task Ties_BreakToTheFirstCandidate()
    {
        // A constant evaluator makes every successor equal; argmax semantics
        // (strict improvement only) must keep the first candidate.
        var state = GameState.NewGame(MatchState.NewMatch(0));
        var plays = MoveGenerator.GeneratePlays(state.Board, die1: 3, die2: 1);
        Assert.True(plays.Count > 1);

        var agent = new OnePlyPlayAgent(
            new TestEvaluators.Constant(new PositionEvaluation(0.5f, 0f, 0f, 0.5f, 0f, 0f)));
        var chosen = await agent.ChoosePlayAsync(state, die1: 3, die2: 1);

        Assert.Equal(plays[0], chosen);
    }

    [Fact]
    public async Task ChosenPlay_IsAlwaysOneOfTheLegalPlays()
    {
        var state = GameState.NewGame(MatchState.NewMatch(0));
        var plays = MoveGenerator.GeneratePlays(state.Board, die1: 6, die2: 5);

        var agent = new OnePlyPlayAgent(new TestEvaluators.PipLead());
        var chosen = await agent.ChoosePlayAsync(state, die1: 6, die2: 5);

        Assert.Contains(chosen, plays);
    }

    [Fact]
    public async Task ChoosePlay_DoesNotMutateTheLiveBoard()
    {
        var state = GameState.NewGame(MatchState.NewMatch(0));
        var before = state.Board.ToMop().ToArray();

        var agent = new OnePlyPlayAgent(new TestEvaluators.PipLead());
        await agent.ChoosePlayAsync(state, die1: 3, die2: 1);

        Assert.Equal(before, state.Board.ToMop().ToArray());
    }

    [Fact]
    public void NullEvaluator_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new OnePlyPlayAgent(null!));
    }

    [Fact]
    public async Task NullState_Throws()
    {
        var agent = new OnePlyPlayAgent(new TestEvaluators.PipLead());
        await Assert.ThrowsAsync<ArgumentNullException>(
            async () => await agent.ChoosePlayAsync(null!, 3, 1));
    }

    [Fact]
    public async Task CanceledToken_Throws()
    {
        var agent = new OnePlyPlayAgent(new TestEvaluators.MustNotBeCalled());
        var state = GameState.NewGame(MatchState.NewMatch(0));

        await Assert.ThrowsAsync<OperationCanceledException>(
            async () => await agent.ChoosePlayAsync(state, 3, 1, new CancellationToken(canceled: true)));
    }
}
