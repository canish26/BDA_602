USE baseball;

CREATE TEMPORARY TABLE IF NOT EXISTS start_pitch_stats ENGINE=MEMORY AS
SELECT
    pc.game_id AS GameID
    , DATE(g.local_date) AS GameDate
    , pc.pitcher AS PitcherID
    , pc.homeTeam AS HomeTeam
    , SUM(pc.plateApperance) AS BattersFaced
    , SUM(pc.outsPlayed/3) AS InningsPitched
    , SUM(pc.Walk)/SUM(pc.outsPlayed)*27 AS BB9
    , SUM(pc.Hit)/SUM(pc.outsPlayed)*27 AS H9
    , SUM(pc.Home_Run)/SUM(pc.outsPlayed)*27 AS HR9
    , SUM(pc.Strikeout)/(SUM(pc.outsPlayed)*3) * 27 AS SO9
    , SUM(pc.Strikeout)/SUM(pc.pitchesThrown) AS StrikeoutPercent
    , 0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run)
        + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk) AS PowerPitchingFactor
    , IF(SUM(pc.outsPlayed) = 0, 0, 3 + ((13*SUM(pc.Home_Run)+3*(SUM(pc.Walk+pc.Hit_By_Pitch)-2*SUM(pc.Strikeout)))/SUM(pc.outsPlayed))) AS DICECoefficient
    , IF(SUM(pc.outsPlayed) = 0, 0, (SUM(pc.Hit)+SUM(pc.Walk))/SUM(pc.outsPlayed)) AS WHIP
    , IF(SUM(pc.outsPlayed)+SUM(pc.plateApperance) = 0, 0, 9*((SUM(pc.Hit)+SUM(pc.Walk)+SUM(pc.Hit_By_Pitch)-SUM(pc.Home_Run))*SUM(pc.pitchesThrown))/(SUM(pc.plateApperance)*SUM(pc.outsPlayed))-0.56) AS CERA
    , SUM(pc.Balk)/SUM(pc.outsPlayed)*27 AS BK9
    , SUM(pc.Hit_By_Pitch)/SUM(pc.outsPlayed)*27 AS HBP9
    , SUM(pc.Wild_Pitch)/SUM(pc.outsPlayed)*27 AS WP9
    , SUM(pc.Intent_Walk)/SUM(pc.plateApperance)*100 AS IBBPercent
    , (SUM(pc.Hit) + SUM(pc.Double) + 2*SUM(pc.Triple) + 3*SUM(pc.Home_Run)) / SUM(pc.plateApperance) AS AVG
    , SUM(pc.Hit)/(SUM(pc.Hit) + SUM(pc.Strikeout)) AS ContactPercent
    , bs.winner_home_or_away AS WinningTeam
    , IF(bs.winner_home_or_away='H', 1, 0) AS HomeTeamWins
FROM game g
    JOIN pitcher_counts pc ON g.game_id = pc.game_id
    JOIN boxscore bs ON g.game_id = bs.game_id
WHERE pc.startingPitcher = 1
GROUP BY GameID, PitcherID
ORDER BY GameID;


ALTER TABLE start_pitch_stats ADD PRIMARY KEY (game_id, pitcher);

CREATE TEMPORARY TABLE IF NOT EXISTS roll_start_pitch_stats ENGINE=MEMORY AS
SELECT
    pc.game_id AS GameID,
    DATE(g.local_date) AS GameDate,
    pc.pitcher AS PitcherID,
    pc.homeTeam AS HomeTeam,
    SUM(pc.plateApperance) AS BattersFaced,
    SUM(pc.outsPlayed/3) AS InningsPitched,
    SUM(pc.Walk)/SUM(pc.outsPlayed)*27 AS BB9,
    SUM(pc.Hit)/SUM(pc.outsPlayed)*27 AS H9,
    SUM(pc.Home_Run)/SUM(pc.outsPlayed)*27 AS HR9,
    SUM(pc.Strikeout)/(SUM(pc.outsPlayed)*3) * 27 AS SO9,
    SUM(pc.Strikeout)/SUM(pc.plateApperance) AS StrikeoutPercent,
    0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run) + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk) AS PowerPitchingFactor,
    IF(SUM(pc.outsPlayed) = 0, 0, (13*SUM(pc.Home_Run) + 3*(SUM(pc.Walk) + SUM(pc.Hit_By_Pitch)) - 2*SUM(pc.Strikeout)) / SUM(pc.outsPlayed) + 3) AS DICECoefficient,
    IF(SUM(pc.outsPlayed) = 0, 0, (SUM(pc.Walk) + SUM(pc.Hit)) / SUM(pc.outsPlayed)) AS WHIP,
    IF(SUM(pc.pitchesThrown) = 0, 0, (SUM(pc.Hit) + SUM(pc.Walk) + SUM(pc.Hit_By_Pitch)) * (0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run) + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk)) / (SUM(pc.pitchesThrown) * SUM(pc.outsPlayed)) * 27 - 0.56) AS CERA,
    SUM(pc.Balk)/SUM(pc.outsPlayed)*27 AS BK9,
    SUM(pc.Hit_By_Pitch)/SUM(pc.outsPlayed)*27 AS HBP9,
    SUM(pc.Wild_Pitch)/SUM(pc.outsPlayed)*27 AS WP9,
    SUM(pc.Intent_Walk)/SUM(pc.plateApperance)*100 AS IBBPercent,
    (SUM(pc.Hit) + SUM(pc.Double) + 2*SUM(pc.Triple) + 3*SUM(pc.Home_Run)) / SUM(pc.plateApperance) AS AVG,
    SUM(pc.Hit)/(SUM(pc.Hit) + SUM(pc.Strikeout)) AS ContactPercent,
    bs.winner_home_or_away AS WinningTeam,
    IF(bs.winner_home_or_away='H', 1, 0) AS HomeTeamWins
FROM game g
    JOIN pitcher_counts pc ON g.game_id = pc.game_id AND pc.startingPitcher = 1
    JOIN boxscore bs ON g.game_id = bs.game_id
GROUP BY GameID, PitcherID
ORDER BY GameID;



ALTER TABLE roll_start_pitch_stats ADD PRIMARY KEY (game_id, pitcher);
