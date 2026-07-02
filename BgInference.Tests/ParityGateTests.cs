namespace BgInference.Tests;

using System.Security.Cryptography;

/// <summary>
/// The cross-language parity gate, in contract order. Every step fails loud
/// and none may be skipped: (1) fixture files exist, (2) the model's sha256
/// matches the vectors header (the two fixtures move together), (3) the
/// model's <c>bgrl.encoding_version</c> matches the encoder (evaluator
/// tests), (4) board→features is bit-exact, (5) features→output is within
/// the fixture's tolerance (evaluator tests). Steps 3 and 5 live with
/// <c>OnnxEvaluator</c>; a red anywhere here means the C# consumer and the
/// Python producer no longer agree and nothing downstream can be trusted.
/// </summary>
public sealed class ParityGateTests
{
    [Fact]
    public void Gate1_FixtureFilesExist()
    {
        Assert.True(
            File.Exists(ParityFixture.ModelPath),
            $"Parity model missing: '{ParityFixture.ModelPath}'. The fixtures are committed in " +
            "BgRLEngine (regenerate with 'python -m parity.generate_vectors'); check the sibling checkout.");
        Assert.True(
            File.Exists(ParityFixture.VectorsPath),
            $"Parity vectors missing: '{ParityFixture.VectorsPath}'. The fixtures are committed in " +
            "BgRLEngine (regenerate with 'python -m parity.generate_vectors'); check the sibling checkout.");
    }

    [Fact]
    public void Gate2_ModelSha256_MatchesVectorsHeader()
    {
        byte[] hash = SHA256.HashData(File.ReadAllBytes(ParityFixture.ModelPath));
        string actual = Convert.ToHexString(hash);

        Assert.True(
            string.Equals(actual, ParityFixture.Vectors.ModelSha256, StringComparison.OrdinalIgnoreCase),
            $"model.onnx sha256 '{actual}' does not match the vectors header " +
            $"'{ParityFixture.Vectors.ModelSha256}'. The two fixture files are a pair and must move " +
            "together — this is a stale or mismatched BgRLEngine checkout, not a tolerance issue.");
    }

    [Fact]
    public void Gate4_VectorsHeader_MatchesEncoderContract()
    {
        // The encoding pin below is only meaningful if the fixture speaks the
        // same contract version and shape this encoder implements.
        Assert.Equal(BgInference.FeatureEncoder.EncodingVersion, ParityFixture.Vectors.EncodingVersion);
        Assert.Equal(BgInference.FeatureEncoder.FeatureSize, ParityFixture.Vectors.InputSize);
        Assert.Equal(0.0, ParityFixture.Vectors.FeatureToleranceAbs); // encoding pin is bit-exact by contract
    }

    public static TheoryData<string> CaseLabels()
    {
        var labels = new TheoryData<string>();
        foreach (var parityCase in ParityFixture.Vectors.Cases)
            labels.Add(parityCase.Label);
        return labels;
    }

    [Theory]
    [MemberData(nameof(CaseLabels))]
    public void Gate4_EncodingPin_BitExact(string label)
    {
        var parityCase = ParityFixture.Case(label);
        var board = ParityFixture.ToBoardState(parityCase.Board);

        Assert.Equal(BgInference.FeatureEncoder.FeatureSize, parityCase.Features.Count);
        var actual = new float[BgInference.FeatureEncoder.FeatureSize];
        BgInference.FeatureEncoder.Encode(
            board,
            parityCase.Board.PlayerToMove,
            parityCase.Board.OffPlayer,
            parityCase.Board.OffOpponent,
            actual);

        double tolerance = ParityFixture.Vectors.FeatureToleranceAbs;
        for (int i = 0; i < actual.Length; i++)
        {
            float expected = (float)parityCase.Features[i];
            if (!(Math.Abs(actual[i] - expected) <= tolerance))
            {
                Assert.Fail(
                    $"Case '{label}': feature[{i}] expected {expected:R}, got {actual[i]:R} " +
                    $"(tolerance {tolerance}). The C# encoder has diverged from encode_board.");
            }
        }
    }
}
