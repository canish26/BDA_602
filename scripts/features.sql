USE baseball;

CREATE TEMPORARY TABLE IF NOT EXISTS start_pitch_stats
ENGINE=MEMORY -- or ENGINE=InnoDB or any other engine you prefer
AS
SELECT
    pc.game_id,
    DATE(g.local_date) AS game_date,
    pc.pitcher,
    pc.homeTeam,
    SUM(pc.outsPlayed/3) AS IP,
    SUM(pc.Walk)/SUM(pc.outsPlayed)*27 AS BB9,
    SUM(pc.Hit)/SUM(pc.outsPlayed)*27 AS HA9,
    SUM(pc.Home_Run)/SUM(pc.outsPlayed)*27 AS HRA9,
    SUM(pc.Strikeout)/SUM(pc.outsPlayed)*27 AS SO9,
    SUM(pc.Strikeout)/SUM(pc.pitchesThrown) AS SOPercent,
    3 + ((13*SUM(pc.Home_Run)+3*(SUM(pc.Walk)+SUM(pc.Hit_By_Pitch)-2*SUM(pc.Strikeout)))/SUM(pc.outsPlayed)) AS DICE,
    (SUM(pc.Walk)+SUM(pc.Hit)) / SUM(pc.outsPlayed) AS WHIP,
    9*((SUM(pc.Hit)+SUM(pc.Walk)+SUM(pc.Hit_By_Pitch))* (1.255*(SUM(pc.Hit)-SUM(pc.Home_Run))+4*SUM(pc.Home_Run))+0.56*(SUM(pc.Walk)+SUM(pc.Hit_By_Pitch)-SUM(pc.Intent_Walk)))/(SUM(pc.outsPlayed)*SUM(pc.plateApperance)) -0.56 AS CERA,
    bs.winner_home_or_away = 'H' AS HTWins
FROM game g
JOIN pitcher_counts pc ON g.game_id = pc.game_id AND pc.startingPitcher = 1
JOIN boxscore bs ON g.game_id = bs.game_id
GROUP BY pc.game_id, pc.pitcher
ORDER BY pc.game_id;

ALTER TABLE start_pitch_stats  ADD PRIMARY KEY (game_id, pitcher);


CREATE TEMPORARY TABLE IF NOT EXISTS roll_start_pitch_stats
ENGINE=MEMORY -- or ENGINE=InnoDB or any other engine you prefer
AS
SELECT
    pc.game_id,
    DATE(g.local_date) AS game_date,
    pc.pitcher,
    pc.homeTeam,
    SUM(pc.outsPlayed)/3.0 AS IP,
    SUM(pc.Walk)/SUM(pc.outsPlayed)*27.0 AS BB9,
    SUM(pc.Hit)/SUM(pc.outsPlayed)*27.0 AS HA9,
    SUM(pc.Home_Run)/SUM(pc.outsPlayed)*27.0 AS HRA9,
    SUM(pc.Strikeout)/SUM(pc.outsPlayed)*27.0 AS SO9,
    SUM(pc.Strikeout)/SUM(pc.pitchesThrown)*100.0 AS SOPercent,
    3.0 + ((13.0*SUM(pc.Home_Run)+3.0*(SUM(pc.Walk)+SUM(pc.Hit_By_Pitch)-2.0*SUM(pc.Strikeout)))/SUM(pc.outsPlayed)) AS DICE,
    (SUM(pc.Walk)+SUM(pc.Hit)) / SUM(pc.outsPlayed) AS WHIP,
    9.0*((SUM(pc.Hit)+SUM(pc.Walk)+SUM(pc.Hit_By_Pitch))* (1.255*(SUM(pc.Hit)-SUM(pc.Home_Run))+4.0*SUM(pc.Home_Run))+0.56*(SUM(pc.Walk)+SUM(pc.Hit_By_Pitch)-SUM(pc.Intent_Walk)))/(SUM(pc.outsPlayed)*SUM(pc.plateApperance)) -0.56 AS CERA
FROM game g
JOIN pitcher_counts pc ON g.game_id = pc.game_id AND pc.startingPitcher = 1
GROUP BY pc.game_id, pc.pitcher
ORDER BY pc.game_id;

ALTER TABLE roll_start_pitch_stats ADD PRIMARY KEY (game_id, pitcher);



CREATE TEMPORARY TABLE start_pitch
ENGINE=MEMORY
AS
SELECT
    h.game_id,
    h.game_date,
    h.pitcher,
    h.homeTeam,
    COALESCE(h.IP, 0) AS IP_HIST,
    COALESCE(h.BB9, 0) AS BB9_HIST,
    COALESCE(h.HA9, 0) AS HA9_HIST,
    COALESCE(h.HRA9, 0) AS HRA9_HIST,
    COALESCE(h.SO9, 0) AS SO9_HIST,
    COALESCE(h.SOPercent, 0) AS SOPercent_HIST,
    COALESCE(h.DICE, 0) AS DICE_HIST,
    COALESCE(h.WHIP, 0) AS WHIP_HIST,
    COALESCE(h.CERA, 0) AS CERA_HIST,
    COALESCE(r.IP, 0) AS IP_ROLL,
    COALESCE(r.BB9, 0) AS BB9_ROLL,
    COALESCE(r.HA9, 0) AS HA9_ROLL,
    COALESCE(r.HRA9, 0) AS HRA9_ROLL,
    COALESCE(r.SO9, 0) AS SO9_ROLL,
    COALESCE(r.SOPercent, 0) AS SOPercent_ROLL,
    COALESCE(r.DICE, 0) AS DICE_ROLL,
    COALESCE(r.WHIP, 0) AS WHIP_ROLL,
    COALESCE(r.CERA, 0) AS CERA_ROLL,
    h.HTWins
FROM start_pitch_stats h
JOIN roll_start_pitch_stats r
    ON h.game_id = r.game_id
    AND h.pitcher = r.pitcher
ORDER BY h.game_id;

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

CREATE TEMPORARY TABLE IF NOT EXISTS team_hist_bat ENGINE=MEMORY AS
SELECT
tbc.game_id,
DATE(g.local_date) AS game_date,
tbc.team_id,
tbc.homeTeam,
tbc.Hit / NULLIF(tbc.atBat, 0) AS AVG,
(tbc.Hit - tbc.Home_Run) / NULLIF((tbc.atBat - tbc.Strikeout - tbc.Home_Run + tbc.Sac_Fly), 0) AS BABIP,
(tbc.Hit + tbc.Walk + tbc.Hit_By_Pitch) / NULLIF((tbc.atBat + tbc.Walk + tbc.Hit_By_Pitch + tbc.Sac_Fly), 0) AS OBP,
(tbc.Single + 2 * tbc.Double + 3 * tbc.Triple + 4 * tbc.Home_Run) / NULLIF(tbc.atBat, 0) AS SLG,
((tbc.Hit + tbc.Walk + tbc.Hit_By_Pitch) / NULLIF((tbc.atBat + tbc.Walk + tbc.Hit_By_Pitch + tbc.Sac_Fly), 0)) +
((tbc.Single + 2 * tbc.Double + 3 * tbc.Triple + 4 * tbc.Home_Run) / NULLIF(tbc.atBat, 0)) AS OPS
FROM game g
JOIN team_batting_counts tbc ON g.game_id = tbc.game_id
ORDER BY tbc.game_id, tbc.team_id;

ALTER TABLE team_hist_bat ADD PRIMARY KEY (game_id, team_id);

CREATE TEMPORARY TABLE IF NOT EXISTS roll_avg_team_bat ENGINE=MEMORY AS (
    SELECT
        tbc.game_id,
        DATE(g.local_date) AS game_date,
        tbc.team_id,
        tbc.homeTeam AS HomeTeam,
        IF(SUM(tbc.atBat)=0, 0, SUM(tbc.Hit)/SUM(tbc.atBat)) AS AVG,
        IF(SUM(tbc.atBat-tbc.Strikeout-tbc.Home_Run+tbc.Sac_Fly)=0, 0, SUM(tbc.Hit-tbc.Home_Run)/SUM(tbc.atBat-tbc.Strikeout-tbc.Home_Run+tbc.Sac_Fly)) AS BABIP,
        IF(SUM(tbc.atBat+tbc.Walk+tbc.Hit_By_Pitch+tbc.Sac_Fly)=0, 0, SUM(tbc.Hit+tbc.Walk+tbc.Hit_By_Pitch)/SUM(tbc.atBat+tbc.Walk+tbc.Hit_By_Pitch+tbc.Sac_Fly)) AS OBP,
        IF(SUM(tbc.atBat)=0, 0, SUM(tbc.Single+(2*tbc.Double)+(3*tbc.Triple)+(4*tbc.Home_Run))/SUM(tbc.atBat)) AS SLG,
        IF(SUM(tbc.atBat)=0 OR SUM(tbc.atBat+tbc.Walk+tbc.Hit_By_Pitch+tbc.Sac_Fly)=0, 0, SUM(tbc.Hit+tbc.Walk+tbc.Hit_By_Pitch)/SUM(tbc.atBat+tbc.Walk+tbc.Hit_By_Pitch+tbc.Sac_Fly) + SUM(tbc.Single+(2*tbc.Double)+(3*tbc.Triple)+(4*tbc.Home_Run))/SUM(tbc.atBat)) AS OPS
    FROM game g
    JOIN team_batting_counts tbc ON g.game_id = tbc.game_id
    GROUP BY tbc.game_id, tbc.team_id
    HAVING game_date > MAX(DATE_SUB(game_date, INTERVAL 100 DAY))
    ORDER BY game_id, team_id
);

ALTER TABLE roll_avg_team_bat ADD PRIMARY KEY (game_id, team_id);

CREATE TEMPORARY TABLE IF NOT EXISTS team_bat ENGINE=MEMORY AS
SELECT
    htb.game_id,
    htb.game_date,
    htb.team_id,
    htb.homeTeam,
    IFNULL(htb.AVG, 0) AS AVG_HIST,
    IFNULL(htb.BABIP, 0) AS BABIP_HIST,
    IFNULL(htb.OBP, 0) AS OBP_HIST,
    IFNULL(htb.SLG, 0) AS SLG_HIST,
    IFNULL(htb.OPS, 0) AS OPS_HIST,
    IFNULL(rtb.AVG, 0) AS AVG_ROLL,
    IFNULL(htb.BABIP, 0) AS BABIP_ROLL,
    IFNULL(rtb.OBP, 0) AS OBP_ROLL,
    IFNULL(rtb.SLG, 0) AS SLG_ROLL,
    IFNULL(rtb.OPS, 0) AS OPS_ROLL
FROM team_hist_bat htb
JOIN roll_avg_team_bat rtb ON htb.game_id = rtb.game_id AND htb.team_id = rtb.team_id
ORDER BY htb.game_id;

ALTER TABLE team_bat ADD PRIMARY KEY (game_id, team_id);

CREATE TEMPORARY TABLE IF NOT EXISTS home_team_bat ENGINE=MEMORY AS
SELECT *
FROM team_bat
WHERE homeTeam=1
ORDER BY game_id;

CREATE TEMPORARY TABLE IF NOT EXISTS away_team_bat ENGINE=MEMORY AS
SELECT *
FROM team_bat
WHERE homeTeam=0
ORDER BY game_id;

ALTER TABLE home_team_bat ADD PRIMARY KEY (game_id, team_id);
ALTER TABLE away_team_bat ADD PRIMARY KEY (game_id, team_id);

CREATE TEMPORARY TABLE IF NOT EXISTS hist_team_pitch ENGINE=MEMORY AS
WITH pitch AS (
    SELECT
        pc.game_id, DATE(g.local_date) AS game_date, pc.team_id, pc.homeTeam AS HomeTeam,
        SUM(pc.Hit) AS H, SUM(pc.atBat) AS AB, SUM(pc.Home_Run) AS HR,
        SUM(pc.Strikeout) AS K, SUM(pc.Sac_Fly) AS SF, SUM(pc.Walk) AS BB,
        SUM(pc.Hit_By_Pitch) AS HBP, SUM(pc.outsPlayed)/3 AS IP,
        SUM(pc.plateApperance) AS BF, SUM(pc.Intent_Walk) AS IBB, SUM(pc.pitchesThrown) AS PT,
        SUM(0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run) + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk)) AS PTB
    FROM game g
    JOIN pitcher_counts pc ON g.game_id = pc.game_id
    GROUP BY pc.game_id, pc.team_id, pc.homeTeam
)
SELECT
    A.game_id, A.game_date, A.team_id, A.HomeTeam,
    SUM(B.IP)/3 AS BFP, SUM(B.IP) AS IP, SUM(B.BB)/SUM(B.IP)*9 AS BB9, SUM(B.H)/SUM(B.IP)*9 AS HA9,
    SUM(B.HR)/SUM(B.IP)*9 AS HRA9, SUM(B.K)/SUM(B.IP)*9 AS SO9,
    SUM(B.K)/SUM(B.PT) AS SOPP,
    IF(SUM(B.IP) = 0, 0, 3 + ((13*SUM(B.HR)+3*(SUM(B.BB)+SUM(B.HBP)-2*B.K))/SUM(B.IP))) AS DICE,
    IF(SUM(B.IP) = 0, 0, (SUM(B.H)+SUM(B.BB)) / SUM(B.IP)) AS WHIP,
    IF(SUM(B.IP)*SUM(B.BF) = 0, 0, 9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))* SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))-0.56) AS CERA
FROM pitch A
LEFT JOIN pitch B ON A.team_id = B.team_id AND A.game_date > B.game_date AND A.game_id = B.game_id
GROUP BY A.HomeTeam, A.game_id, A.team_id, B.K
ORDER BY A.game_id, A.team_id;


ALTER TABLE hist_team_pitch ADD PRIMARY KEY (game_id, team_id);

#SHOWING MEMORY IS FULL
CREATE TEMPORARY TABLE roll_team_pitch
AS (
    WITH pitcher_dummy AS (
        SELECT
            pc.game_id, DATE(g.local_date) AS game_date, pc.team_id, pc.homeTeam AS HomeTeam,
            SUM(pc.Hit) AS H, SUM(pc.atBat) AS AB, SUM(pc.Home_Run) AS HR, SUM(pc.Strikeout) AS K,
            SUM(pc.Sac_Fly) AS SF, SUM(pc.Walk) AS BB, SUM(pc.Hit_By_Pitch) AS HBP,
            SUM(pc.outsPlayed)/3 AS IP, SUM(pc.plateApperance) AS BF, SUM(pc.Intent_Walk) AS IBB,
            SUM(pc.pitchesThrown) AS PT,
            SUM(0.89*(1.255*(pc.Hit-pc.Home_Run) + 4*pc.Home_Run) + 0.56*(pc.Walk+pc.Hit_By_Pitch-pc.Intent_Walk)) AS PTB
        FROM game g JOIN pitcher_counts pc ON g.game_id = pc.game_id
        GROUP BY pc.game_id, pc.team_id, pc.homeTeam
    )
    SELECT A.game_id, A.game_date, A.team_id, A.HomeTeam, SUM(B.BF)/SUM(B.IP) AS BFP, SUM(B.IP) AS IP,
           SUM(B.BB)/9 AS BB9, SUM(B.H)/9 AS HA9, SUM(B.HR)/9 AS HRA9, SUM(B.K)/(9*SUM(B.IP)) AS SO9,
           SUM(B.K)/SUM(B.PT) AS SOPP,
           IF(SUM(B.IP)=0,0,3+((13*SUM(B.HR)+3*(SUM(B.BB+B.HBP)-2*B.K))/SUM(B.IP))) AS DICE,
           IF(SUM(B.IP)=0,0,(SUM(B.H)+SUM(B.BB))/SUM(B.IP)) AS WHIP,
           IF(SUM(B.IP)*SUM(B.BF)=0,0,9*((SUM(B.H)+SUM(B.BB)+SUM(B.HBP))*SUM(B.PTB))/(SUM(B.BF)*SUM(B.IP))-0.56) AS CERA
    FROM pitcher_dummy A
    LEFT JOIN pitcher_dummy B ON A.team_id = B.team_id AND A.game_date > B.game_date
        AND B.game_date >= DATE_SUB(A.game_date, INTERVAL 100 DAY)
    GROUP BY A.game_id, A.team_id, A.HomeTeam, B.K
    ORDER BY A.game_id, A.team_id
);


ALTER TABLE roll_team_pitch ADD PRIMARY KEY (game_id, team_id);

-- Create a temporary table called team_pitch using the MEMORY engine
-- This table will contain aggregated pitching statistics for each team across multiple games
CREATE TEMPORARY TABLE team_pitch
AS
SELECT
    htp.game_id,
    htp.game_date,
    htp.team_id,
    htp.homeTeam,
    COALESCE(htp.BFP, 0) AS BFP_HIST,
    COALESCE(htp.IP, 0) AS IP_HIST,
    COALESCE(htp.BB9, 0) AS BB9_HIST,
    COALESCE(htp.HA9, 0) AS HA9_HIST,
    COALESCE(htp.HRA9, 0) AS HRA9_HIST,
    COALESCE(htp.SO9, 0) AS SO9_HIST,
    COALESCE(rtp.IP, 0) AS IP_ROLL,
    COALESCE(rtp.BB9, 0) AS BB9_ROLL,
    COALESCE(rtp.HA9, 0) AS HA9_ROLL,
    COALESCE(rtp.HRA9, 0) AS HRA9_ROLL,
    COALESCE(rtp.SO9, 0) AS SO9_ROLL,
    COALESCE(rtp.SOPP, 0) AS SOPP_ROLL,
    COALESCE(rtp.DICE, 0) AS DICE_ROLL,
    COALESCE(rtp.WHIP, 0) AS WHIP_ROLL,
    COALESCE(htp.SOPP, 0) AS SOPercent_HIST,
    COALESCE(htp.DICE, 0) AS DICE_HIST,
    COALESCE(htp.WHIP, 0) AS WHIP_HIST,
    COALESCE(htp.CERA, 0) AS CERA_HIST,
    COALESCE(rtp.BFP, 0) AS BFP_ROLL,

    COALESCE(rtp.CERA, 0) AS CERA_ROLL
FROM hist_team_pitch htp
JOIN roll_team_pitch rtp ON htp.game_id = rtp.game_id AND htp.team_id = rtp.team_id
ORDER BY htp.game_id;

-- Add a primary key constraint to the game_id and team_id columns
ALTER TABLE team_pitch ADD PRIMARY KEY (game_id, team_id);

CREATE TEMPORARY TABLE home_team_pitch AS
SELECT * FROM team_pitch WHERE homeTeam=1;

CREATE INDEX idx_home_team_pitch ON home_team_pitch (game_id, team_id);

CREATE TEMPORARY TABLE away_team_pitch AS
SELECT * FROM team_pitch WHERE homeTeam=0;

CREATE INDEX idx_away_team_pitch ON away_team_pitch (game_id, team_id);

CREATE TABLE IF NOT EXISTS features_start_pitch AS
SELECT
    hsp.game_id
    , hsp.game_date
    , COALESCE(hsp.BFP_HIST,0) - COALESCE(asp.BFP_HIST,0) AS SP_BFP_DIFF_HIST
    , COALESCE(hsp.IP_HIST,0) - COALESCE(asp.IP_HIST,0) AS SP_IP_DIFF_HIST
    , COALESCE(hsp.BB9_HIST,0) - COALESCE(asp.BB9_HIST,0) AS SP_BB9_DIFF_HIST
    , COALESCE(hsp.HA9_HIST,0) - COALESCE(asp.HA9_HIST,0) AS SP_HA9_DIFF_HIST
    , COALESCE(hsp.HRA9_HIST,0) - COALESCE(asp.HRA9_HIST,0) AS SP_HRA9_DIFF_HIST
    , COALESCE(hsp.SO9_HIST,0) - COALESCE(asp.SO9_HIST,0) AS SP_SO9_DIFF_HIST
    , COALESCE(hsp.BFP_ROLL,0) - COALESCE(asp.BFP_ROLL,0) AS SP_BFP_DIFF_ROLL
    , COALESCE(hsp.IP_ROLL,0) - COALESCE(asp.IP_ROLL,0) AS SP_IP_DIFF_ROLL
    , COALESCE(hsp.BB9_ROLL,0) - COALESCE(asp.BB9_ROLL,0) AS SP_BB9_DIFF_ROLL
    , COALESCE(hsp.HA9_ROLL,0) - COALESCE(asp.HA9_ROLL,0) AS SP_HA9_DIFF_ROLL
    , COALESCE(hsp.HRA9_ROLL,0) - COALESCE(asp.HRA9_ROLL,0) AS SP_HRA9_DIFF_ROLL
    , COALESCE(hsp.SO9_ROLL,0) - COALESCE(asp.SO9_ROLL,0) AS SP_SO9_DIFF_ROLL
    , COALESCE(hsp.SOPercent_ROLL,0) - COALESCE(asp.SOPercent_ROLL,0) AS SP_SOPP_DIFF_ROLL
    , COALESCE(hsp.DICE_ROLL,0) - COALESCE(asp.DICE_ROLL,0) AS SP_DICE_DIFF_ROLL
    , COALESCE(hsp.WHIP_ROLL,0) - COALESCE(asp.WHIP_ROLL,0) AS SP_WHIP_DIFF_ROLL
    , COALESCE(hsp.SOPercent_HIST,0) - COALESCE(asp.SOPercent_HIST,0) AS SP_SOPP_DIFF_HIST
    , COALESCE(hsp.DICE_HIST,0) - COALESCE(asp.DICE_HIST,0) AS SP_DICE_DIFF_HIST
    , COALESCE(hsp.WHIP_HIST,0) - COALESCE(asp.WHIP_HIST,0) AS SP_WHIP_DIFF_HIST
    , COALESCE(hsp.CERA_HIST,0) - COALESCE(asp.CERA_HIST,0) AS SP_CERA_DIFF_HIST
    , COALESCE(hsp.CERA_ROLL,0) - COALESCE(asp.CERA_ROLL,0) AS SP_CERA_DIFF_ROLL
    , hsp.HTWins
FROM home_start_pitch hsp JOIN away_start_pitch asp
    ON hsp.game_id = asp.game_id
;

ALTER TABLE features_start_pitch ADD PRIMARY KEY (game_id, game_date);

CREATE TABLE IF NOT EXISTS features_team_bat AS
SELECT
    htb.game_id
    , htb.game_date
    , COALESCE(htb.AVG_HIST, 0) - COALESCE(atb.AVG_HIST, 0) AS TB_AVG_DIFF_HIST
    , COALESCE(htb.BABIP_HIST, 0) - COALESCE(atb.BABIP_HIST, 0) AS TB_BABIP_DIFF_HIST
    , COALESCE(htb.OBP_HIST, 0) - COALESCE(atb.OBP_HIST, 0) AS TB_OBP_DIFF_HIST
    , COALESCE(htb.SLG_HIST, 0) - COALESCE(atb.SLG_HIST, 0) AS TB_SLG_DIFF_HIST
    , COALESCE(htb.OPS_HIST, 0) - COALESCE(atb.SLG_HIST, 0) AS TB_OPS_DIFF_HIST
    , COALESCE(htb.AVG_ROLL, 0) - COALESCE(atb.AVG_ROLL, 0) AS TB_AVG_DIFF_ROLL
    , COALESCE(htb.BABIP_ROLL, 0) - COALESCE(atb.BABIP_ROLL, 0) AS TB_BABIP_DIFF_ROLL
    , COALESCE(htb.OBP_ROLL, 0) - COALESCE(atb.OBP_ROLL, 0) AS TB_OBP_DIFF_ROLL
    , COALESCE(htb.SLG_ROLL, 0) - COALESCE(atb.SLG_ROLL, 0) AS TB_SLG_DIFF_ROLL
    , COALESCE(htb.OPS_ROLL, 0) - COALESCE(atb.OPS_ROLL, 0) AS TB_OPS_DIFF_ROLL
FROM home_team_bat htb JOIN away_team_bat atb
    ON htb.game_id = atb.game_id
;

ALTER TABLE features_team_bat ADD PRIMARY KEY (game_id, game_date);


CREATE TABLE IF NOT EXISTS features_team_pitch AS
SELECT
    htp.game_id
    , htp.game_date
    , COALESCE(htp.BFP_HIST, 0) - COALESCE(atp.BFP_HIST, 0) AS TP_BFP_DIFF_HIST
    , COALESCE(htp.IP_HIST, 0) - COALESCE(atp.IP_HIST, 0) AS TP_IP_DIFF_HIST
    , COALESCE(htp.BB9_HIST, 0) - COALESCE(atp.BB9_HIST, 0) AS TP_BB9_DIFF_HIST
    , COALESCE(htp.HA9_HIST, 0) - COALESCE(atp.HA9_HIST, 0) AS TP_HA9_DIFF_HIST
    , COALESCE(htp.HRA9_HIST, 0) - COALESCE(atp.HRA9_HIST, 0) AS TP_HRA9_DIFF_HIST
    , COALESCE(htp.SO9_HIST, 0) - COALESCE(atp.SO9_HIST, 0) AS TP_SO9_DIFF_HIST
    , COALESCE(htp.SOPercent_HIST, 0) - COALESCE(atp.SOPercent_HIST, 0) AS TP_SOPP_DIFF_HIST
    , COALESCE(htp.DICE_HIST, 0) - COALESCE(atp.DICE_HIST, 0) AS TP_DICE_DIFF_HIST
    , COALESCE(htp.WHIP_HIST, 0) - COALESCE(atp.WHIP_HIST, 0) AS TP_WHIP_DIFF_HIST
    , COALESCE(htp.CERA_HIST, 0) - COALESCE(atp.CERA_HIST, 0) AS TP_CERA_DIFF_HIST
    , COALESCE(htp.BFP_ROLL, 0) - COALESCE(atp.BFP_ROLL, 0) AS TP_BFP_DIFF_ROLL
    , COALESCE(htp.IP_ROLL, 0) - COALESCE(atp.IP_ROLL, 0) AS TP_IP_DIFF_ROLL
    , COALESCE(htp.BB9_ROLL, 0) - COALESCE(atp.BB9_ROLL, 0) AS TP_BB9_DIFF_ROLL
    , COALESCE(htp.HA9_ROLL, 0) - COALESCE(atp.HA9_ROLL, 0) AS TP_HA9_DIFF_ROLL
    , COALESCE(htp.HRA9_ROLL, 0) - COALESCE(atp.HRA9_ROLL, 0) AS TP_HRA9_DIFF_ROLL
    , COALESCE(htp.SO9_ROLL, 0) - COALESCE(atp.SO9_ROLL, 0) AS TP_SO9_DIFF_ROLL
    , COALESCE(htp.SOPP_ROLL, 0) - COALESCE(atp.SOPP_ROLL, 0) AS TP_SOPP_DIFF_ROLL
    , COALESCE(htp.DICE_ROLL, 0) - COALESCE(atp.DICE_ROLL, 0) AS TP_DICE_DIFF_ROLL
    , COALESCE(htp.WHIP_ROLL, 0) - COALESCE(atp.WHIP_ROLL, 0) AS TP_WHIP_DIFF_ROLL
    , COALESCE(htp.CERA_ROLL, 0) - COALESCE(atp.CERA_ROLL, 0) AS TP_CERA_DIFF_ROLL
FROM home_team_pitch htp JOIN away_team_pitch atp
    ON htp.game_id = atp.game_id
;

ALTER TABLE features_team_pitch ADD PRIMARY KEY (game_id, game_date);



