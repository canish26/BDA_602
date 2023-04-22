USE baseball;

CREATE TEMPORARY TABLE IF NOT EXISTS start_pitch_stats ENGINE=MEMORY AS
SELECT
    pc.game_id AS GameID
    , DATE(g.date) AS GameDate
    , pc.pitcher AS PitcherID
    , pc.homeTeam AS HomeTeam
    , SUM(pc.plateAppearance) AS BattersFaced
    , SUM(pc.outs/3) AS InningsPitched
    , SUM(pc.walk)/SUM(pc.outs)*27 AS BB9
    , SUM(pc.hit)/SUM(pc.outs)*27 AS H9
    , SUM(pc.homeRun)/SUM(pc.outs)*27 AS HR9
    , SUM(pc.strikeout)/(SUM(pc.outs)*3) * 27 AS SO9
    , SUM(pc.strikeout)/SUM(pc.totalPitches) AS StrikeoutPercent
    , 0.89*(1.255*(pc.hit-pc.homeRun) + 4*pc.homeRun)
        + 0.56*(pc.walk+pc.hitByPitch-pc.intentionalWalk) AS PowerPitchingFactor
    , IF(SUM(pc.outs) = 0, 0, 3 + ((13*SUM(pc.homeRun)+3*(SUM(pc.walk+pc.hitByPitch)-2*SUM(pc.strikeout)))/SUM(pc.outs))) AS DICECoefficient
    , IF(SUM(pc.outs) = 0, 0, (SUM(pc.hit)+SUM(pc.walk))/SUM(pc.outs)) AS WHIP
    , IF(SUM(pc.outs)+SUM(pc.plateAppearance) = 0, 0, 9*((SUM(pc.hit)+SUM(pc.walk)+SUM(pc.hitByPitch)-SUM(pc.homeRun))*SUM(pc.totalPitches))/(SUM(pc.plateAppearance)*SUM(pc.outs))-0.56) AS CERA
    , SUM(pc.balk)/SUM(pc.outs)*27 AS BK9
    , SUM(pc.hitByPitch)/SUM(pc.outs)*27 AS HBP9
    , SUM(pc.wildPitch)/SUM(pc.outs)*27 AS WP9
    , SUM(pc.intentionalWalk)/SUM(pc.plateAppearance)*100 AS IBBPercent
    , (SUM(pc.hit) + SUM(pc.double) + 2*SUM(pc.triple) + 3*SUM(pc.homeRun)) / SUM(pc.plateAppearance) AS AVG
    , SUM(pc.hit)/(SUM(pc.hit) + SUM(pc.strikeout)) AS ContactPercent
    , bs.winningTeam AS WinningTeam
    , IF(bs.winningTeam='home', 1, 0) AS HomeTeamWins
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
    IF(bs.winner_home_or_away='H', 1, 0) AS HomeTeamWins,
    (SUM(pc.Hit) + SUM(pc.Double)*2 + SUM(pc.Triple)*3 + SUM(pc.Home_Run)*4) / SUM(pc.plateApperance) AS TotalBases,
    (SUM(pc.Hit) + SUM(pc.Double)*2 + SUM(pc.Triple)*3 + SUM(pc.Home_Run)*4) / SUM(pc.At_Bats) AS SluggingPercentage
FROM game g
    JOIN pitcher_counts pc ON g.game_id = pc.game_id AND pc.startingPitcher = 1
    JOIN boxscore bs ON g.game_id = bs.game_id
GROUP BY GameID, PitcherID
ORDER BY GameID;


ALTER TABLE roll_start_pitch_stats ADD PRIMARY KEY (game_id, pitcher);

CREATE TEMPORARY TABLE IF NOT EXISTS start_pitch ENGINE=MEMORY AS
SELECT
    hsp.GameID AS game_id,
    hsp.GameDate AS game_date,
    hsp.PitcherID AS pitcher_id,
    hsp.HomeTeam AS home_team,
    IF(hsp.BattersFaced IS NULL, 0, hsp.BattersFaced) AS batters_faced_hist,
    IF(hsp.InningsPitched IS NULL, 0, hsp.InningsPitched) AS innings_pitched_hist,
    IF(hsp.BB9 IS NULL, 0, hsp.BB9) AS bb9_hist,
    IF(hsp.H9 IS NULL, 0, hsp.H9) AS ha9_hist,
    IF(hsp.HR9 IS NULL, 0, hsp.HR9) AS hra9_hist,
    IF(hsp.SO9 IS NULL, 0, hsp.SO9) AS so9_hist,
    IF(hsp.StrikeoutPercent IS NULL, 0, hsp.StrikeoutPercent) AS so_percent_hist,
    IF(hsp.DICECoefficient IS NULL, 0, hsp.DICECoefficient) AS dice_hist,
    IF(hsp.WHIP IS NULL, 0, hsp.WHIP) AS whip_hist,
    IF(hsp.CERA IS NULL, 0, hsp.CERA) AS cera_hist,
    IF(rsp.BattersFaced IS NULL, 0, rsp.BattersFaced) AS batters_faced_roll,
    IF(rsp.InningsPitched IS NULL, 0, rsp.InningsPitched) AS innings_pitched_roll,
    IF(rsp.BB9 IS NULL, 0, rsp.BB9) AS bb9_roll,
    IF(rsp.H9 IS NULL, 0, rsp.H9) AS ha9_roll,
    IF(rsp.HR9 IS NULL, 0, rsp.HR9) AS hra9_roll,
    IF(rsp.SO9 IS NULL, 0, rsp.SO9) AS so9_roll,
    IF(rsp.StrikeoutPercent IS NULL, 0, rsp.StrikeoutPercent) AS so_percent_roll,
    IF(rsp.DICECoefficient IS NULL, 0, rsp.DICECoefficient) AS dice_roll,
    IF(rsp.WHIP IS NULL, 0, rsp.WHIP) AS whip_roll,
    IF(rsp.CERA IS NULL, 0, rsp.CERA) AS cera_roll,
    hsp.HomeTeamWins AS home_team_wins
FROM (
    SELECT *
    FROM roll_start_pitch_stats
) hsp
LEFT JOIN (
    SELECT *
    FROM start_pitch_stats
) rsp ON hsp.GameID = rsp.GameID AND hsp.PitcherID = rsp.PitcherID
ORDER BY hsp.GameID;



ALTER TABLE start_pitch ADD PRIMARY KEY (game_id, pitcher);

CREATE TEMPORARY TABLE home_start_pitch ENGINE=MEMORY AS
SELECT *
FROM start_pitch
WHERE homeTeam = 1
ORDER BY game_id, pitcher;

CREATE TEMPORARY TABLE away_start_pitch ENGINE=MEMORY AS
SELECT *
FROM start_pitch
WHERE homeTeam = 0
ORDER BY game_id, pitcher;

ALTER TABLE home_start_pitch ADD PRIMARY KEY (game_id, pitcher);
ALTER TABLE away_start_pitch ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE IF NOT EXISTS feature_data AS
(
SELECT
hsp.game_id,
hsp.game_date,
hsp.BFP_HIST - asp.BFP_HIST AS BFP_HIST_DIFF,
hsp.IP_HIST - asp.IP_HIST AS IP_HIST_DIFF,
hsp.BB9_HIST - asp.BB9_HIST AS BB9_HIST_DIFF,
hsp.HA9_HIST - asp.HA9_HIST AS HA9_HIST_DIFF,
hsp.HRA9_HIST - asp.HRA9_HIST AS HRA9_HIST_DIFF,
hsp.SO9_HIST - asp.SO9_HIST AS SO9_HIST_DIFF,
hsp.SOPercent_HIST - asp.SOPercent_HIST AS SOPercent_HIST_DIFF,
hsp.DICE_HIST - asp.DICE_HIST AS DICE_HIST_DIFF,
hsp.WHIP_HIST - asp.WHIP_HIST AS WHIP_HIST_DIFF,
hsp.CERA_HIST - asp.CERA_HIST AS CERA_HIST_DIFF,
hsp.BFP_ROLL - asp.BFP_ROLL AS BFP_ROLL_DIFF,
hsp.IP_ROLL - asp.IP_ROLL AS IP_ROLL_DIFF,
hsp.BB9_ROLL - asp.BB9_ROLL AS BB9_ROLL_DIFF,
hsp.HA9_ROLL - asp.HA9_ROLL AS HA9_ROLL_DIFF,
hsp.HRA9_ROLL - asp.HRA9_ROLL AS HRA9_ROLL_DIFF,
hsp.SO9_ROLL - asp.SO9_ROLL AS SO9_ROLL_DIFF,
hsp.SOPercent_ROLL - asp.SOPercent_ROLL AS SOPercent_ROLL_DIFF,
hsp.DICE_ROLL - asp.DICE_ROLL AS DICE_ROLL_DIFF,
hsp.WHIP_ROLL - asp.WHIP_ROLL AS WHIP_ROLL_DIFF,
hsp.CERA_ROLL - asp.CERA_ROLL AS CERA_ROLL_DIFF,
hsp.HTWins,
hsp.BattersFaced_HIST - asp.BattersFaced_HIST AS BattersFaced_HIST_DIFF,
hsp.InningsPitched_HIST - asp.InningsPitched_HIST AS InningsPitched_HIST_DIFF,
hsp.BattersFaced_ROLL - asp.BattersFaced_ROLL AS BattersFaced_ROLL_DIFF,
hsp.InningsPitched_ROLL - asp.InningsPitched_ROLL AS InningsPitched_ROLL_DIFF,
hsp.BB9_30 - asp.BB9_30 AS BB9_30_DIFF,
hsp.HA9_30 - asp.HA9_30 AS HA9_30_DIFF,
hsp.HRA9_30 - asp.HRA9_30 AS HRA9_30_DIFF,
hsp.SO9_30 - asp.SO9_30 AS SO9_30_DIFF,
hsp.SOPercent_30 - asp.SOPercent_30 AS SOPercent_30_DIFF,
hsp.DICE_30 - asp.DICE_30 AS DICE_30_DIFF,
hsp.WHIP_30 - asp.WHIP_30 AS WHIP_30_DIFF,
hsp.CERA_30 - asp.CERA_30 AS CERA_30_DIFF
FROM
start_pitch hsp
INNER JOIN
start_pitch asp
ON hsp.game_id = asp.game_id AND hsp.homeTeam != asp.homeTeam
);
