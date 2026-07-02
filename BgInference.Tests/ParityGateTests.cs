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

    [Fact]
    public void Gate3_MetadataHandshake_ParityModelLoads()
    {
        // Load() performs the handshake; reaching the assertions means it passed.
        var evaluator = ParityFixture.Evaluator;

        Assert.Equal("1", evaluator.ModelMetadata["bgrl.encoding_version"]);
        Assert.Equal("303", evaluator.ModelMetadata["bgrl.input_size"]);
        Assert.Equal("6", evaluator.ModelMetadata["bgrl.num_outputs"]);
        Assert.Equal("parity", evaluator.ModelMetadata["bgrl.model_role"]);
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

    [Theory]
    [MemberData(nameof(CaseLabels))]
    public void Gate5_InferencePin_FeaturesToOutput(string label)
    {
        var parityCase = ParityFixture.Case(label);
        var features = parityCase.Features.Select(d => (float)d).ToArray();

        var evaluation = ParityFixture.Evaluator.RunInference(features, count: 1)[0];

        AssertOutputWithinTolerance(label, parityCase, evaluation);
    }

    [Fact]
    public void Gate5_InferencePin_HoldsForTheWholeBatchInOneRun()
    {
        // Same pin, all 28 rows in a single model invocation — the dynamic
        // batch dimension must not change any row's result.
        var cases = ParityFixture.Vectors.Cases;
        var features = new float[cases.Count * BgInference.FeatureEncoder.FeatureSize];
        for (int i = 0; i < cases.Count; i++)
        {
            for (int j = 0; j < BgInference.FeatureEncoder.FeatureSize; j++)
                features[(i * BgInference.FeatureEncoder.FeatureSize) + j] = (float)cases[i].Features[j];
        }

        var evaluations = ParityFixture.Evaluator.RunInference(features, cases.Count);

        for (int i = 0; i < cases.Count; i++)
            AssertOutputWithinTolerance(cases[i].Label, cases[i], evaluations[i]);
    }

    [Fact]
    public void PublicPipeline_BoardToOutput_MatchesFixtureForOnRollCases()
    {
        // Ties the whole public path (BoardState → derived offs → features →
        // inference) to the fixture. Only on-roll cases apply: the public
        // evaluator always encodes player-to-move, by design.
        var cases = ParityFixture.Vectors.Cases.Where(c => c.Board.PlayerToMove).ToList();
        Assert.NotEmpty(cases);

        var boards = cases.Select(c => ParityFixture.ToBoardState(c.Board)).ToList();
        var evaluations = ParityFixture.Evaluator.EvaluateBatch(boards);

        for (int i = 0; i < cases.Count; i++)
            AssertOutputWithinTolerance(cases[i].Label, cases[i], evaluations[i]);
    }

    private static void AssertOutputWithinTolerance(
        string label, ParityCase parityCase, BgInference.PositionEvaluation evaluation)
    {
        Assert.Equal(6, parityCase.Output.Count);
        float[] actual =
        [
            evaluation.PWin, evaluation.PWinGammon, evaluation.PWinBackgammon,
            evaluation.PLose, evaluation.PLoseGammon, evaluation.PLoseBackgammon,
        ];

        double tolerance = ParityFixture.Vectors.OutputToleranceAbs;
        for (int j = 0; j < actual.Length; j++)
        {
            float expected = (float)parityCase.Output[j];
            if (!(Math.Abs(actual[j] - expected) <= tolerance))
            {
                Assert.Fail(
                    $"Case '{label}': output[{j}] expected {expected:R}, got {actual[j]:R} " +
                    $"(tolerance {tolerance}). C# inference has diverged from the producer's ONNX Runtime run.");
            }
        }
    }
}
