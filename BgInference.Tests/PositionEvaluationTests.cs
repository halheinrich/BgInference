namespace BgInference.Tests;

using BgInference;

/// <summary>Equity-fold arithmetic and the money-weights pin.</summary>
public sealed class PositionEvaluationTests
{
    [Fact]
    public void MoneyWeights_MirrorComputeEquityDefault()
    {
        // Pinned to the producer's compute_equity money default. Note this is
        // NOT the cumulative-probability fold (1,1,1,−1,−1,−1): the producer
        // treats the six outputs as exclusive outcome classes.
        Assert.Equal(new EquityWeights(1f, 2f, 3f, -1f, -2f, -3f), EquityWeights.Money);
    }

    [Theory]
    [InlineData(1f, 0f, 0f, 0f, 0f, 0f, 1f)]   // single win → +1
    [InlineData(0f, 1f, 0f, 0f, 0f, 0f, 2f)]   // gammon win → +2
    [InlineData(0f, 0f, 1f, 0f, 0f, 0f, 3f)]   // backgammon win → +3
    [InlineData(0f, 0f, 0f, 1f, 0f, 0f, -1f)]  // single loss → −1
    [InlineData(0f, 0f, 0f, 0f, 1f, 0f, -2f)]  // gammon loss → −2
    [InlineData(0f, 0f, 0f, 0f, 0f, 1f, -3f)]  // backgammon loss → −3
    public void Equity_WeightsEachOutcomeClass(
        float pWin, float pWinG, float pWinB, float pLose, float pLoseG, float pLoseB, float expected)
    {
        var evaluation = new PositionEvaluation(pWin, pWinG, pWinB, pLose, pLoseG, pLoseB);
        Assert.Equal(expected, evaluation.Equity(EquityWeights.Money));
    }

    [Fact]
    public void Equity_IsTheDotProduct()
    {
        var evaluation = new PositionEvaluation(0.5f, 0.25f, 0.125f, 0.5f, 0.25f, 0.125f);
        float expected = (0.5f * 1f) + (0.25f * 2f) + (0.125f * 3f)
            + (0.5f * -1f) + (0.25f * -2f) + (0.125f * -3f);
        Assert.Equal(expected, evaluation.Equity(EquityWeights.Money));
    }
}
