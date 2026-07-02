namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgInference;

/// <summary>
/// Evaluator behavior around the parity gate: the fail-fast handshake
/// (end-to-end through <see cref="OnnxEvaluator.Load(string)"/> against real
/// files, plus per-violation validator units), the public evaluation surface,
/// and lifetime.
/// </summary>
public sealed class OnnxEvaluatorTests
{
    private static Dictionary<string, string> ValidMetadata() => new()
    {
        ["bgrl.encoding_version"] = "1",
        ["bgrl.input_size"] = "303",
        ["bgrl.num_outputs"] = "6",
    };

    // ── Handshake, end-to-end through Load ───────────────────────────

    [Fact]
    public void Load_MissingFile_ThrowsFileNotFound()
    {
        var missing = Path.Combine(ParityFixture.ParityDir, "no-such-model.onnx");
        Assert.Throws<FileNotFoundException>(() => OnnxEvaluator.Load(missing));
    }

    [Fact]
    public void Load_NonOnnxFile_ThrowsModelContractException()
    {
        // vectors.json exists but is not an ONNX model — the load failure must
        // surface as the named contract exception, with the native cause inner.
        var ex = Assert.Throws<ModelContractException>(
            () => OnnxEvaluator.Load(ParityFixture.VectorsPath));

        Assert.Contains("not a loadable ONNX model", ex.Message, StringComparison.Ordinal);
        Assert.NotNull(ex.InnerException);
    }

    [Fact]
    public void Load_EncodingVersionMismatch_ThrowsModelContractException_EndToEnd()
    {
        // The real parity model through the real Load path, with only the
        // required version doctored: proves Load itself runs the handshake —
        // a validator-only unit test would not catch Load forgetting to call it.
        var ex = Assert.Throws<ModelContractException>(
            () => OnnxEvaluator.Load(ParityFixture.ModelPath, requiredEncodingVersion: 999));

        Assert.Contains("bgrl.encoding_version", ex.Message, StringComparison.Ordinal);
        Assert.Contains("999", ex.Message, StringComparison.Ordinal);
    }

    // ── Handshake, per-violation validator units ─────────────────────

    [Fact]
    public void ValidateMetadata_MissingEncodingVersion_Throws()
    {
        var metadata = ValidMetadata();
        metadata.Remove("bgrl.encoding_version");

        var ex = Assert.Throws<ModelContractException>(
            () => OnnxEvaluator.ValidateMetadata(metadata, 1, "test.onnx"));
        Assert.Contains("bgrl.encoding_version", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateMetadata_NonIntegerEncodingVersion_Throws()
    {
        var metadata = ValidMetadata();
        metadata["bgrl.encoding_version"] = "1.0";

        var ex = Assert.Throws<ModelContractException>(
            () => OnnxEvaluator.ValidateMetadata(metadata, 1, "test.onnx"));
        Assert.Contains("not an integer", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateMetadata_WrongInputSize_Throws()
    {
        var metadata = ValidMetadata();
        metadata["bgrl.input_size"] = "290";

        var ex = Assert.Throws<ModelContractException>(
            () => OnnxEvaluator.ValidateMetadata(metadata, 1, "test.onnx"));
        Assert.Contains("bgrl.input_size", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateMetadata_WrongNumOutputs_Throws()
    {
        var metadata = ValidMetadata();
        metadata["bgrl.num_outputs"] = "5";

        var ex = Assert.Throws<ModelContractException>(
            () => OnnxEvaluator.ValidateMetadata(metadata, 1, "test.onnx"));
        Assert.Contains("bgrl.num_outputs", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ValidateMetadata_ValidContract_DoesNotThrow()
    {
        OnnxEvaluator.ValidateMetadata(ValidMetadata(), 1, "test.onnx");
    }

    // ── Evaluation surface ───────────────────────────────────────────

    [Fact]
    public void Evaluate_MatchesBatchRow()
    {
        var board = BoardState.Standard();

        var single = ParityFixture.Evaluator.Evaluate(board);
        var batch = ParityFixture.Evaluator.EvaluateBatch([board, BoardState.Nackgammon()]);

        Assert.Equal(batch[0], single);
        Assert.NotEqual(batch[1], single); // different position, different row
    }

    [Fact]
    public void EvaluateBatch_Empty_ReturnsEmpty()
    {
        Assert.Empty(ParityFixture.Evaluator.EvaluateBatch([]));
    }

    [Fact]
    public void EvaluateBatch_NullList_Throws()
    {
        Assert.Throws<ArgumentNullException>(
            () => ParityFixture.Evaluator.EvaluateBatch(null!));
    }

    [Fact]
    public void Evaluate_NullBoard_Throws()
    {
        Assert.Throws<ArgumentNullException>(
            () => ParityFixture.Evaluator.Evaluate(null!));
    }

    [Fact]
    public void EvaluateBatch_NullElement_ThrowsNamingTheIndex()
    {
        var ex = Assert.Throws<ArgumentException>(
            () => ParityFixture.Evaluator.EvaluateBatch([BoardState.Standard(), null!]));
        Assert.Contains("boards[1]", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EvaluateBatch_MoreThanFifteenCheckers_FailsLoud()
    {
        var mop = new int[26];
        mop[1] = 16; // pseudoboard: off-count derivation would go negative
        var board = BoardState.FromMop(mop);

        var ex = Assert.Throws<ArgumentException>(
            () => ParityFixture.Evaluator.Evaluate(board));
        Assert.Contains("more than 15 checkers", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Evaluate_GameOverBoard_Evaluates()
    {
        // All checkers off on both sides — degenerate but representable; the
        // evaluator must tolerate it (match loops can reach it), not throw.
        var board = BoardState.FromMop(new int[26]);
        ParityFixture.Evaluator.Evaluate(board);
    }

    // ── Lifetime ─────────────────────────────────────────────────────

    [Fact]
    public void Dispose_ThenEvaluate_ThrowsObjectDisposed()
    {
        var evaluator = OnnxEvaluator.Load(ParityFixture.ModelPath);
        evaluator.Dispose();

        Assert.Throws<ObjectDisposedException>(
            () => evaluator.Evaluate(BoardState.Standard()));
    }

    [Fact]
    public void Dispose_Twice_IsIdempotent()
    {
        var evaluator = OnnxEvaluator.Load(ParityFixture.ModelPath);
        evaluator.Dispose();
        evaluator.Dispose();
    }
}
