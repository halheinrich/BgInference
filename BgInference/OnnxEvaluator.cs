namespace BgInference;

using System.Collections.ObjectModel;
using System.Globalization;
using BgDataTypes_Lib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// Position evaluator over a BgRLEngine-exported ONNX model (CPU inference
/// via ONNX Runtime).
///
/// <para>
/// <see cref="Load(string)"/> performs the fail-fast contract handshake
/// before any evaluation is possible: the file must be loadable ONNX, its
/// <c>bgrl.encoding_version</c> metadata must equal the encoding version this
/// library implements, the structural metadata (<c>bgrl.input_size</c>,
/// <c>bgrl.num_outputs</c>) must match, and the graph must be
/// <c>features</c> float32 <c>[batch, 303]</c> → <c>probabilities</c> float32
/// <c>[batch, 6]</c>. Any violation throws <see cref="ModelContractException"/>
/// — a model that would silently mis-evaluate must never finish loading.
/// </para>
///
/// <para>
/// Instances are safe for concurrent <see cref="Evaluate"/> /
/// <see cref="EvaluateBatch"/> calls (ONNX Runtime sessions are thread-safe
/// for inference). Dispose releases the native session.
/// </para>
/// </summary>
public sealed class OnnxEvaluator : IPositionEvaluator, IDisposable
{
    private const string InputName = "features";
    private const string OutputName = "probabilities";
    private const string EncodingVersionKey = "bgrl.encoding_version";
    private const string InputSizeKey = "bgrl.input_size";
    private const string NumOutputsKey = "bgrl.num_outputs";
    private const int NumOutputs = 6;

    private readonly InferenceSession _session;
    private bool _disposed;

    private OnnxEvaluator(InferenceSession session, IReadOnlyDictionary<string, string> metadata)
    {
        _session = session;
        ModelMetadata = metadata;
    }

    /// <summary>
    /// The model's embedded <c>bgrl.*</c> metadata (encoding version,
    /// structural keys, role, checkpoint provenance where present) — already
    /// handshake-validated; exposed for diagnostics and routing.
    /// </summary>
    public IReadOnlyDictionary<string, string> ModelMetadata { get; }

    /// <summary>
    /// Load a BgRLEngine-exported model and perform the contract handshake.
    /// </summary>
    /// <param name="modelPath">Path to the exported <c>.onnx</c> file.</param>
    /// <returns>A ready evaluator owning the inference session.</returns>
    /// <exception cref="FileNotFoundException">The file does not exist.</exception>
    /// <exception cref="ModelContractException">The file violates the export contract; the message names the specific violation.</exception>
    public static OnnxEvaluator Load(string modelPath) =>
        Load(modelPath, FeatureEncoder.EncodingVersion);

    /// <summary>
    /// <see cref="Load(string)"/> with the required encoding version as a
    /// parameter — the seam that lets tests drive the version-mismatch
    /// handshake end-to-end against a real model file.
    /// </summary>
    internal static OnnxEvaluator Load(string modelPath, int requiredEncodingVersion)
    {
        ArgumentNullException.ThrowIfNull(modelPath);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"ONNX model not found: '{modelPath}'.", modelPath);

        InferenceSession? session = null;
        try
        {
            try
            {
                session = new InferenceSession(modelPath);
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                throw new ModelContractException(
                    $"'{modelPath}' is not a loadable ONNX model.", ex);
            }

            var metadata = new ReadOnlyDictionary<string, string>(
                new Dictionary<string, string>(session.ModelMetadata.CustomMetadataMap));
            ValidateMetadata(metadata, requiredEncodingVersion, modelPath);
            ValidateGraph(session, modelPath);

            var evaluator = new OnnxEvaluator(session, metadata);
            session = null; // ownership transferred
            return evaluator;
        }
        finally
        {
            session?.Dispose();
        }
    }

    /// <summary>
    /// The metadata half of the handshake, isolated so each violation is unit-
    /// testable without fabricating ONNX files: keys must exist, parse as
    /// integers, and match the encoder's contract.
    /// </summary>
    internal static void ValidateMetadata(
        IReadOnlyDictionary<string, string> metadata,
        int requiredEncodingVersion,
        string modelPath)
    {
        int encodingVersion = RequireIntKey(metadata, EncodingVersionKey, modelPath);
        if (encodingVersion != requiredEncodingVersion)
        {
            throw new ModelContractException(
                $"'{modelPath}' has {EncodingVersionKey} = {encodingVersion}, but this library implements " +
                $"encoding version {requiredEncodingVersion}. The model was exported against an incompatible " +
                "feature encoding; re-export it with a matching BgRLEngine, or update this library.");
        }

        int inputSize = RequireIntKey(metadata, InputSizeKey, modelPath);
        if (inputSize != FeatureEncoder.FeatureSize)
        {
            throw new ModelContractException(
                $"'{modelPath}' has {InputSizeKey} = {inputSize}; expected {FeatureEncoder.FeatureSize}.");
        }

        int numOutputs = RequireIntKey(metadata, NumOutputsKey, modelPath);
        if (numOutputs != NumOutputs)
        {
            throw new ModelContractException(
                $"'{modelPath}' has {NumOutputsKey} = {numOutputs}; expected {NumOutputs}.");
        }
    }

    private static int RequireIntKey(
        IReadOnlyDictionary<string, string> metadata, string key, string modelPath)
    {
        if (!metadata.TryGetValue(key, out string? raw))
        {
            throw new ModelContractException(
                $"'{modelPath}' has no '{key}' metadata — not a BgRLEngine export " +
                "(the bgrl.* contract keys are embedded by engine/export.py).");
        }

        if (!int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out int value))
        {
            throw new ModelContractException(
                $"'{modelPath}' has '{key}' = '{raw}', which is not an integer.");
        }

        return value;
    }

    /// <summary>
    /// The graph half of the handshake: one float32 input <c>features</c>
    /// <c>[batch, 303]</c>, one float32 output <c>probabilities</c>
    /// <c>[batch, 6]</c>.
    /// </summary>
    private static void ValidateGraph(InferenceSession session, string modelPath)
    {
        if (!session.InputMetadata.TryGetValue(InputName, out var input))
        {
            throw new ModelContractException(
                $"'{modelPath}' has no graph input named '{InputName}' " +
                $"(inputs: {string.Join(", ", session.InputMetadata.Keys)}).");
        }

        if (input.ElementType != typeof(float)
            || input.Dimensions.Length != 2
            || input.Dimensions[1] != FeatureEncoder.FeatureSize)
        {
            throw new ModelContractException(
                $"'{modelPath}' input '{InputName}' is not float32 [batch, {FeatureEncoder.FeatureSize}] " +
                $"(got {input.ElementType.Name} [{string.Join(", ", input.Dimensions)}]).");
        }

        if (!session.OutputMetadata.TryGetValue(OutputName, out var output))
        {
            throw new ModelContractException(
                $"'{modelPath}' has no graph output named '{OutputName}' " +
                $"(outputs: {string.Join(", ", session.OutputMetadata.Keys)}).");
        }

        if (output.ElementType != typeof(float)
            || output.Dimensions.Length != 2
            || output.Dimensions[1] != NumOutputs)
        {
            throw new ModelContractException(
                $"'{modelPath}' output '{OutputName}' is not float32 [batch, {NumOutputs}] " +
                $"(got {output.ElementType.Name} [{string.Join(", ", output.Dimensions)}]).");
        }
    }

    /// <inheritdoc />
    public PositionEvaluation Evaluate(BoardState board)
    {
        ArgumentNullException.ThrowIfNull(board);
        return EvaluateBatch([board])[0];
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">A board's derived borne-off count is negative (more than 15 checkers a side).</exception>
    public PositionEvaluation[] EvaluateBatch(IReadOnlyList<BoardState> boards)
    {
        ArgumentNullException.ThrowIfNull(boards);
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (boards.Count == 0)
            return [];

        var features = new float[boards.Count * FeatureEncoder.FeatureSize];
        for (int i = 0; i < boards.Count; i++)
        {
            var board = boards[i] ?? throw new ArgumentException($"boards[{i}] is null.", nameof(boards));
            (int offPlayer, int offOpponent) = DeriveOffCounts(board, i);
            FeatureEncoder.Encode(
                board,
                playerToMove: true,
                offPlayer,
                offOpponent,
                features.AsSpan(i * FeatureEncoder.FeatureSize, FeatureEncoder.FeatureSize));
        }

        return RunInference(features, boards.Count);
    }

    /// <summary>
    /// The raw features→output half of the pipeline — the seam the parity
    /// gate's inference pin drives with the fixture's own feature rows,
    /// isolating model inference from board encoding.
    /// </summary>
    /// <param name="features">Row-major <c>[count, 303]</c> feature rows.</param>
    /// <param name="count">Number of rows.</param>
    internal PositionEvaluation[] RunInference(float[] features, int count)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        var tensor = new DenseTensor<float>(features, [count, FeatureEncoder.FeatureSize]);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(InputName, tensor) };

        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == OutputName).AsTensor<float>();

        var evaluations = new PositionEvaluation[count];
        for (int i = 0; i < count; i++)
        {
            evaluations[i] = new PositionEvaluation(
                output[i, 0], output[i, 1], output[i, 2],
                output[i, 3], output[i, 4], output[i, 5]);
        }

        return evaluations;
    }

    /// <summary>
    /// Borne-off counts, derived: <see cref="BoardState"/> stores board and
    /// bar checkers only, so each side's offs are 15 minus what remains in
    /// play. A negative derivation means more than 15 checkers a side — not a
    /// backgammon position, and an evaluation of it would be silently
    /// meaningless, so it fails loud instead.
    /// </summary>
    private static (int OffPlayer, int OffOpponent) DeriveOffCounts(BoardState board, int index)
    {
        int player = 0;
        int opponent = 0;
        for (int p = 1; p <= 24; p++)
        {
            int n = board.Points[p];
            if (n > 0) player += n;
            else opponent -= n;
        }
        player += board.Points[25];
        opponent -= board.Points[0];

        int offPlayer = FeatureEncoder.CheckersPerPlayer - player;
        int offOpponent = FeatureEncoder.CheckersPerPlayer - opponent;
        if (offPlayer < 0 || offOpponent < 0)
        {
            throw new ArgumentException(
                $"boards[{index}] has more than {FeatureEncoder.CheckersPerPlayer} checkers a side " +
                $"(on-roll: {player}, opponent: {opponent}) — not an evaluable backgammon position.");
        }

        return (offPlayer, offOpponent);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _session.Dispose();
    }
}
