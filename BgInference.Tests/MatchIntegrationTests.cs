namespace BgInference.Tests;

using BgGame_Lib;
using BgInference;

/// <summary>
/// The engine-track integration proof: the ONNX agent pair (one-ply play
/// over the parity model + threshold cube) enters a real seeded match via
/// <c>MatchRunner</c> and plays it to completion. The parity model plays
/// legally-but-weakly by design — strength is not a gate; a trained model
/// slots in by pointing <see cref="OnnxEvaluator.Load(string)"/> at a
/// BgRLEngine <c>export_onnx.py</c> artifact instead of the parity fixture.
/// Assertions are shape-level (completion, score consistency), not
/// transcript-exact: <c>SeededDiceSource</c> is stable per .NET runtime
/// version, not across them.
/// </summary>
public sealed class MatchIntegrationTests
{
    private static MatchParticipant OnnxParticipant()
    {
        var evaluator = ParityFixture.Evaluator;
        return new MatchParticipant(
            new OnePlyPlayAgent(evaluator), new ThresholdCubeAgent(evaluator));
    }

    private static MatchParticipant TrivialParticipant() =>
        new(new FirstPlayAgent(), new NeverCubeAgent());

    private static void AssertMatchCompleted(MatchResult result, int matchLength)
    {
        Assert.NotNull(result.Winner);
        int winnerScore = result.Winner == MatchSeat.One ? result.SeatOneScore : result.SeatTwoScore;
        int loserScore = result.Winner == MatchSeat.One ? result.SeatTwoScore : result.SeatOneScore;
        Assert.True(winnerScore >= matchLength,
            $"Winner score {winnerScore} did not reach the match length {matchLength}.");
        Assert.True(loserScore < matchLength,
            $"Loser score {loserScore} reached the match length {matchLength} too.");

        Assert.NotEmpty(result.Games);
        Assert.All(result.Games, game =>
        {
            Assert.True(game.Result.Points >= 1, "Every completed game scores at least one point.");
            Assert.NotEmpty(game.Transcript.Entries);
        });
        Assert.Equal(result.SeatOneScore + result.SeatTwoScore, result.Games.Sum(g => g.Result.Points));
    }

    [Fact]
    public async Task OnnxAgents_PlayAFullSeededMatch_ToCompletion()
    {
        var runner = new MatchRunner(new SeededDiceSource(seed: 20260702));

        var result = await runner.RunMatchAsync(
            OnnxParticipant(), TrivialParticipant(), matchLength: 3, maxGames: 50);

        AssertMatchCompleted(result, matchLength: 3);
    }

    [Fact]
    public async Task OnnxAgents_AlsoCompleteFromSeatTwo()
    {
        // Same proof, other chair: no hidden seat dependence in the agents.
        var runner = new MatchRunner(new SeededDiceSource(seed: 7));

        var result = await runner.RunMatchAsync(
            TrivialParticipant(), OnnxParticipant(), matchLength: 3, maxGames: 50);

        AssertMatchCompleted(result, matchLength: 3);
    }
}
