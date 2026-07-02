namespace BgInference;

/// <summary>
/// Thrown when a model file violates the BgRLEngine export contract — it is
/// not loadable ONNX, its <c>bgrl.*</c> metadata is missing or malformed, its
/// <c>bgrl.encoding_version</c> does not match the version this library's
/// encoder implements, or its graph shape is not the contract's
/// <c>features [batch, 303] → probabilities [batch, 6]</c>. This is the
/// fail-fast handshake: a model that would silently mis-evaluate must never
/// finish loading.
/// </summary>
public sealed class ModelContractException : Exception
{
    /// <summary>Create with a message describing the specific contract violation.</summary>
    public ModelContractException(string message)
        : base(message)
    {
    }

    /// <summary>Create with a message and the underlying failure.</summary>
    public ModelContractException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}
