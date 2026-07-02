namespace BgInference.Tests;

using BgDataTypes_Lib;
using BgInference;

/// <summary>
/// Unit tests for the encoder's individual feature groups. The authoritative
/// whole-vector check is the parity gate (<see cref="ParityGateTests"/>);
/// these pin each group's arithmetic and layout offsets so a divergence
/// diagnoses to a specific feature family rather than "vector differs".
/// </summary>
public sealed class FeatureEncoderTests
{
    // Layout offsets (from the producer's encode_board layout).
    private const int OpponentPointsBase = 144;
    private const int BarPlayerBase = 288;
    private const int BarOpponentBase = 291;
    private const int OffPlayerBase = 294;
    private const int OffOpponentBase = 296;
    private const int PlayerToMoveIndex = 298;
    private const int PipRatioIndex = 299;
    private const int RaceIndex = 300;
    private const int PlayerCountIndex = 301;
    private const int OpponentCountIndex = 302;

    private static float[] Encode(int[] mop, bool playerToMove = true, int offPlayer = 0, int offOpponent = 0)
    {
        var features = new float[FeatureEncoder.FeatureSize];
        FeatureEncoder.Encode(BoardState.FromMop(mop), playerToMove, offPlayer, offOpponent, features);
        return features;
    }

    private static int[] EmptyMop() => new int[26];

    public static TheoryData<int, float[]> PointEncodings() => new()
    {
        { 0, new[] { 0f, 0f, 0f, 0f, 0f, 0f } },
        { 1, new[] { 1f, 0f, 0f, 0f, 0f, 0f } },
        { 3, new[] { 1f, 1f, 1f, 0f, 0f, 0f } },
        { 5, new[] { 1f, 1f, 1f, 1f, 1f, 0f } },
        { 6, new[] { 1f, 1f, 1f, 1f, 1f, 0.1f } },        // overflow: (6−5)/10
        { 15, new[] { 1f, 1f, 1f, 1f, 1f, 1f } },          // overflow saturates at 1.0
        { 16, new[] { 1f, 1f, 1f, 1f, 1f, 1f } },          // clamped, not 1.1
    };

    [Theory]
    [MemberData(nameof(PointEncodings))]
    public void PlayerPoint_ThermometerPlusOverflow(int count, float[] expected)
    {
        var mop = EmptyMop();
        mop[3] = count; // player point 3 → features[(3−1)×6 ..]
        var features = Encode(mop);

        Assert.Equal(expected, features[12..18]);
    }

    [Fact]
    public void OpponentPoint_EncodesInOpponentBlock_NotPlayerBlock()
    {
        var mop = EmptyMop();
        mop[7] = -4; // opponent's 4 checkers on point 7
        var features = Encode(mop);

        int opponentSlot = OpponentPointsBase + (7 - 1) * 6;
        Assert.Equal(new[] { 1f, 1f, 1f, 1f, 0f, 0f }, features[opponentSlot..(opponentSlot + 6)]);
        Assert.Equal(new[] { 0f, 0f, 0f, 0f, 0f, 0f }, features[36..42]); // player slot for point 7 stays empty
    }

    [Theory]
    [InlineData(0, 0f, 0f, 0f)]
    [InlineData(1, 1f, 0f, 0f)]
    [InlineData(2, 1f, 1f, 0f)]
    [InlineData(3, 1f, 1f, 1f)]
    [InlineData(4, 1f, 1f, 1f)] // depth-3 thermometer caps
    public void PlayerBar_Thermometer(int count, float unit0, float unit1, float unit2)
    {
        var mop = EmptyMop();
        mop[25] = count;
        var features = Encode(mop);

        Assert.Equal(new[] { unit0, unit1, unit2 }, features[BarPlayerBase..(BarPlayerBase + 3)]);
    }

    [Fact]
    public void OpponentBar_ReadFromNegatedPointsZero()
    {
        var mop = EmptyMop();
        mop[0] = -2; // opponent bar is stored ≤ 0
        var features = Encode(mop);

        Assert.Equal(new[] { 1f, 1f, 0f }, features[BarOpponentBase..(BarOpponentBase + 3)]);
        Assert.Equal(new[] { 0f, 0f, 0f }, features[BarPlayerBase..(BarPlayerBase + 3)]);
    }

    [Theory]
    [InlineData(0, 0f, 0f)]
    [InlineData(7, 7f / 15f, 0f)]
    [InlineData(15, 1f, 1f)]
    public void BorneOff_FractionAndAllOffFlag(int off, float fraction, float allOff)
    {
        var features = Encode(EmptyMop(), offPlayer: off, offOpponent: 0);

        Assert.Equal(new[] { fraction, allOff }, features[OffPlayerBase..(OffPlayerBase + 2)]);
    }

    [Fact]
    public void OpponentBorneOff_EncodesInItsOwnSlot()
    {
        var features = Encode(EmptyMop(), offPlayer: 0, offOpponent: 15);

        Assert.Equal(new[] { 0f, 0f }, features[OffPlayerBase..(OffPlayerBase + 2)]);
        Assert.Equal(new[] { 1f, 1f }, features[OffOpponentBase..(OffOpponentBase + 2)]);
    }

    [Fact]
    public void PlayerToMove_FlipsExactlyOneFeature()
    {
        var mop = BoardState.Standard().ToMop().ToArray();
        var onRoll = Encode(mop, playerToMove: true);
        var notOnRoll = Encode(mop, playerToMove: false);

        Assert.Equal(1f, onRoll[PlayerToMoveIndex]);
        Assert.Equal(0f, notOnRoll[PlayerToMoveIndex]);
        for (int i = 0; i < FeatureEncoder.FeatureSize; i++)
        {
            if (i != PlayerToMoveIndex)
                Assert.Equal(onRoll[i], notOnRoll[i]);
        }
    }

    [Fact]
    public void PipRatio_EmptyBoard_IsHalf()
    {
        Assert.Equal(0.5f, Encode(EmptyMop())[PipRatioIndex]);
    }

    [Fact]
    public void PipRatio_IsPlayerShareOfTotalPips()
    {
        var mop = EmptyMop();
        mop[1] = 2;   // player: 2 checkers × 1 pip = 2
        mop[24] = -1; // opponent: 1 checker × (25 − 24) = 1
        var features = Encode(mop);

        Assert.Equal((float)(2.0 / 3.0), features[PipRatioIndex]);
    }

    [Fact]
    public void Race_ContactPosition_IsZero()
    {
        var mop = BoardState.Standard().ToMop().ToArray();
        Assert.Equal(0f, Encode(mop)[RaceIndex]);
    }

    [Fact]
    public void Race_DisjointRanges_IsOne()
    {
        var mop = EmptyMop();
        mop[3] = 2;   // player near bear-off
        mop[20] = -2; // opponent past all player checkers
        Assert.Equal(1f, Encode(mop)[RaceIndex]);
    }

    [Fact]
    public void Race_PlayerOnBar_IsZero_EvenWhenIsRaceIsVacuouslyTrue()
    {
        // Producer's is_race() returns false for ANY occupied bar; with no
        // opponent checkers anywhere, BoardState.IsRace calls the same
        // position vacuously true. The encoder must pin the producer's answer.
        var mop = EmptyMop();
        mop[25] = 1;
        var board = BoardState.FromMop(mop);

        Assert.True(board.IsRace); // documents the divergence this guard exists for
        Assert.Equal(0f, Encode(mop)[RaceIndex]);
    }

    [Fact]
    public void CheckerCounts_IncludeBar_NormalizedByFifteen()
    {
        var mop = EmptyMop();
        mop[6] = 4;   // player: 4 on board
        mop[25] = 1;  //         + 1 on bar = 5
        mop[19] = -3; // opponent: 3 on board
        var features = Encode(mop);

        Assert.Equal((float)(5 / 15.0), features[PlayerCountIndex]);
        Assert.Equal((float)(3 / 15.0), features[OpponentCountIndex]);
    }

    [Fact]
    public void Encode_WrongDestinationLength_Throws()
    {
        var board = BoardState.Standard();
        var tooShort = new float[FeatureEncoder.FeatureSize - 1];

        Assert.Throws<ArgumentException>(
            () => FeatureEncoder.Encode(board, playerToMove: true, 0, 0, tooShort));
    }
}
