namespace BgInference.Tests;

using System.Text.Json;
using System.Text.Json.Serialization;
using BgDataTypes_Lib;

/// <summary>
/// Locates and loads the cross-language parity fixtures committed in the
/// BgRLEngine sibling submodule: <c>parity/model.onnx</c> (tiny deterministic
/// model) and <c>parity/vectors.json</c> (28 golden board→features→output
/// triples, sha256-paired to the model). Consumed in place — never copied —
/// so the fixtures have exactly one source of truth; the gate's sha256 step
/// makes a stale pairing impossible.
/// </summary>
internal static class ParityFixture
{
    /// <summary>
    /// Sibling-submodule fixture directory, reached from the test assembly's
    /// output directory: five levels up to the umbrella checkout root, then
    /// into BgRLEngine (same convention as the shared <c>TestData/</c> path
    /// in sibling test projects).
    /// </summary>
    internal static string ParityDir { get; } = Path.GetFullPath(
        Path.Combine(AppContext.BaseDirectory, @"..\..\..\..\..\BgRLEngine\BgRLEngine\parity"));

    internal static string ModelPath => Path.Combine(ParityDir, "model.onnx");

    internal static string VectorsPath => Path.Combine(ParityDir, "vectors.json");

    private static readonly Lazy<ParityVectors> _vectors = new(() =>
        JsonSerializer.Deserialize<ParityVectors>(File.ReadAllText(VectorsPath))
            ?? throw new InvalidDataException($"'{VectorsPath}' deserialized to null."));

    /// <summary>The parsed vectors file (header + all golden cases).</summary>
    internal static ParityVectors Vectors => _vectors.Value;

    private static readonly Lazy<OnnxEvaluator> _evaluator = new(() =>
        OnnxEvaluator.Load(ModelPath));

    /// <summary>
    /// A shared evaluator over the parity model. Deliberately never disposed:
    /// one tiny native session for the whole test run, released at process
    /// exit. Tests that exercise disposal load their own instance.
    /// </summary>
    internal static OnnxEvaluator Evaluator => _evaluator.Value;

    /// <summary>Find a golden case by its unique label.</summary>
    internal static ParityCase Case(string label) =>
        Vectors.Cases.Single(c => c.Label == label);

    /// <summary>
    /// Build a <see cref="BoardState"/> from a fixture board's raw fields.
    /// The producer's <c>points[i]</c> (index 0 = player's 1-point) is
    /// <c>Points[i + 1]</c> here — same sign convention (positive = the
    /// perspective player), same orientation; bars map to <c>Points[25]</c>
    /// (player, ≥ 0) and <c>Points[0]</c> (opponent, stored ≤ 0). Off counts
    /// and the player-to-move flag are not part of <see cref="BoardState"/>
    /// and travel separately.
    /// </summary>
    internal static BoardState ToBoardState(ParityBoard board)
    {
        var mop = new int[26];
        mop[0] = -board.BarOpponent;
        for (int i = 0; i < 24; i++)
            mop[i + 1] = board.Points[i];
        mop[25] = board.BarPlayer;
        return BoardState.FromMop(mop);
    }
}

/// <summary>Deserialized <c>vectors.json</c>: header plus golden cases.</summary>
internal sealed record ParityVectors
{
    [JsonPropertyName("format_version")] public int FormatVersion { get; init; }
    [JsonPropertyName("encoding_version")] public int EncodingVersion { get; init; }
    [JsonPropertyName("input_size")] public int InputSize { get; init; }
    [JsonPropertyName("num_outputs")] public int NumOutputs { get; init; }
    [JsonPropertyName("output_semantics")] public string OutputSemantics { get; init; } = "";
    [JsonPropertyName("model_file")] public string ModelFile { get; init; } = "";
    [JsonPropertyName("model_sha256")] public string ModelSha256 { get; init; } = "";
    [JsonPropertyName("feature_tolerance_abs")] public double FeatureToleranceAbs { get; init; }
    [JsonPropertyName("output_tolerance_abs")] public double OutputToleranceAbs { get; init; }
    [JsonPropertyName("cases")] public IReadOnlyList<ParityCase> Cases { get; init; } = [];
}

/// <summary>One golden triple: a raw board, its features, and the model's output.</summary>
internal sealed record ParityCase
{
    [JsonPropertyName("label")] public string Label { get; init; } = "";
    [JsonPropertyName("board")] public ParityBoard Board { get; init; } = new();
    [JsonPropertyName("features")] public IReadOnlyList<double> Features { get; init; } = [];
    [JsonPropertyName("output")] public IReadOnlyList<double> Output { get; init; } = [];
}

/// <summary>
/// A fixture board in the producer's raw shape: <c>points[24]</c> with index 0
/// = the perspective player's 1-point (positive = theirs), explicit bar and
/// borne-off counts per side, and the player-to-move flag.
/// </summary>
internal sealed record ParityBoard
{
    [JsonPropertyName("points")] public IReadOnlyList<int> Points { get; init; } = [];
    [JsonPropertyName("bar_player")] public int BarPlayer { get; init; }
    [JsonPropertyName("bar_opponent")] public int BarOpponent { get; init; }
    [JsonPropertyName("off_player")] public int OffPlayer { get; init; }
    [JsonPropertyName("off_opponent")] public int OffOpponent { get; init; }
    [JsonPropertyName("player_to_move")] public bool PlayerToMove { get; init; }
}
