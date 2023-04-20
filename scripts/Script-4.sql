/*Batting Average (BA): */
SELECT
  game.home_team_id AS team_id,
  'Home' AS home_away,
  SUM(batter_counts.Hit) / SUM(batter_counts.atBat) AS batter_counts_avg
FROM
  game
  JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.home_team_id = batter_counts.team_id
GROUP BY
  game.home_team_id, 'Home'

UNION ALL

SELECT
  game.away_team_id AS team_id,
  'Away' AS home_away,
  SUM(batter_counts.Hit) / SUM(batter_counts.atBat) AS batter_counts_avg
FROM
  game
  JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.away_team_id = batter_counts.team_id
GROUP BY
  game.away_team_id, 'Away';
  
/*On-Base Percentage (OBP):  */
 
 SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((SUM(batter_counts.Hit)+SUM(batter_counts.Walk)+SUM(batter_counts.Hit_By_Pitch))/
          (SUM(batter_counts.atBat)+SUM(batter_counts.Walk)+SUM(batter_counts.Hit_By_Pitch)+SUM(batter_counts.Sac_Fly)),3) AS home_obp,
    NULL AS away_obp
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.home_team_id = batter_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_obp,
    ROUND((SUM(batter_counts.Hit)+SUM(batter_counts.Walk)+SUM(batter_counts.Hit_By_Pitch))/
          (SUM(batter_counts.atBat)+SUM(batter_counts.Walk)+SUM(batter_counts.Hit_By_Pitch)+SUM(batter_counts.Sac_Fly)),3) AS away_obp
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.away_team_id = batter_counts.team_id
GROUP BY game.game_id, game.away_team_id;


/*Slugging Percentage (SLG):*/
SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((SUM(batter_counts.Single) + 2 * SUM(batter_counts.Double) + 3 * SUM(batter_counts.Triple) + 4 * SUM(batter_counts.Home_Run)) / SUM(batter_counts.atBat),3) AS home_slg,
    NULL AS away_slg
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.home_team_id = batter_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_slg,
    ROUND((SUM(batter_counts.Single) + 2 * SUM(batter_counts.Double) + 3 * SUM(batter_counts.Triple) + 4 * SUM(batter_counts.Home_Run)) / SUM(batter_counts.atBat),3) AS away_slg
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.away_team_id = batter_counts.team_id
GROUP BY game.game_id, game.away_team_id;

/*On-Base Plus Slugging (OPS)*/

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND(((SUM(batter_counts.Hit) + SUM(batter_counts.Walk) + SUM(batter_counts.Hit_By_Pitch)) / (SUM(batter_counts.atBat) + SUM(batter_counts.Walk) + SUM(batter_counts.Hit_By_Pitch) + SUM(batter_counts.Sac_Fly))) 
    + ((SUM(batter_counts.Single) + 2 * SUM(batter_counts.Double) + 3 * SUM(batter_counts.Triple) + 4 * SUM(batter_counts.Home_Run)) / SUM(batter_counts.atBat)),3) AS home_ops,
    NULL AS away_ops
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.home_team_id = batter_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_ops,
    ROUND(((SUM(batter_counts.Hit) + SUM(batter_counts.Walk) + SUM(batter_counts.Hit_By_Pitch)) / (SUM(batter_counts.atBat) + SUM(batter_counts.Walk) + SUM(batter_counts.Hit_By_Pitch) + SUM(batter_counts.Sac_Fly))) 
    + ((SUM(batter_counts.Single) + 2 * SUM(batter_counts.Double) + 3 * SUM(batter_counts.Triple) + 4 * SUM(batter_counts.Home_Run)) / SUM(batter_counts.atBat)),3) AS away_ops
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.away_team_id = batter_counts.team_id
GROUP BY game.game_id, game.away_team_id;

/*Runs Created (RC):*/

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((SUM(batter_counts.Hit) + SUM(batter_counts.Walk)) * (SUM(batter_counts.Hit) + SUM(batter_counts.Double) * 2 + SUM(batter_counts.Triple) * 3 + SUM(batter_counts.Home_Run) * 4) / (SUM(batter_counts.atBat) + SUM(batter_counts.Walk)), 3) AS home_rc,
    NULL AS away_rc
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.home_team_id = batter_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_rc,
    ROUND((SUM(batter_counts.Hit) + SUM(batter_counts.Walk)) * (SUM(batter_counts.Hit) + SUM(batter_counts.Double) * 2 + SUM(batter_counts.Triple) * 3 + SUM(batter_counts.Home_Run) * 4) / (SUM(batter_counts.atBat) + SUM(batter_counts.Walk)), 3) AS away_rc
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.away_team_id = batter_counts.team_id
GROUP BY game.game_id, game.away_team_id;


/*Strikeout-to-Walk Ratio (K/BB)*/

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND(SUM(pitcher_counts.Strikeout) / SUM(pitcher_counts.Walk), 2) AS home_k_bb_ratio,
    NULL AS away_k_bb_ratio
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.home_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_k_bb_ratio,
    ROUND(SUM(pitcher_counts.Strikeout) / SUM(pitcher_counts.Walk), 2) AS away_k_bb_ratio
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.away_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.away_team_id;

/*WHIP (Walks plus Hits per Inning Pitched)*/

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((SUM(pitcher_counts.Walk) + SUM(pitcher_counts.Hit)) / (SUM(pitcher_counts.Strikeout) / 3), 3) AS home_whip,
    NULL AS away_whip
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.home_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_whip,
    ROUND((SUM(pitcher_counts.Walk) + SUM(pitcher_counts.Hit)) / (SUM(pitcher_counts.Strikeout) / 3), 3) AS away_whip
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.away_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.away_team_id;


/*CERA (Component ERA)*/
SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((13 * SUM(pitcher_counts.Home_Run) + 3 * (SUM(pitcher_counts.Walk) + SUM(pitcher_counts.Hit_By_Pitch)) - 2 * SUM(pitcher_counts.Strikeout)) / SUM(pitcher_counts.pitchesThrown) * 27, 2) AS home_cera,
    NULL AS away_cera
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.home_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_cera,
    ROUND((13 * SUM(pitcher_counts.Home_Run) + 3 * (SUM(pitcher_counts.Walk) + SUM(pitcher_counts.Hit_By_Pitch)) - 2 * SUM(pitcher_counts.Strikeout)) / SUM(pitcher_counts.pitchesThrown) * 27, 2) AS away_cera
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.away_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.away_team_id;

/*Strikeouts per Nine Innings (K/9)*/
SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((SUM(pitcher_counts.Strikeout)/SUM(pitcher_counts.pitchesThrown)*9), 1) AS home_k_per_9,
    NULL AS away_k_per_9
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.home_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_k_per_9,
    ROUND((SUM(pitcher_counts.Strikeout)/SUM(pitcher_counts.pitchesThrown)*9), 1) AS away_k_per_9
FROM game 
JOIN pitcher_counts ON game.game_id = pitcher_counts.game_id AND game.away_team_id = pitcher_counts.team_id
GROUP BY game.game_id, game.away_team_id;

/*BABIP – Batting average on balls in play*/
SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Home' AS home_away,
    ROUND((SUM(batter_counts.Hit)-SUM(batter_counts.Home_Run))/
          (SUM(batter_counts.atBat)-SUM(batter_counts.Strikeout)-SUM(batter_counts.Home_Run)+SUM(batter_counts.Sac_Fly)),3) AS home_babip,
    NULL AS away_babip
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.home_team_id = batter_counts.team_id
GROUP BY game.game_id, game.home_team_id

UNION ALL

SELECT 
    game.game_id, 
    game.home_team_id, 
    game.away_team_id,
    'Away' AS home_away,
    NULL AS home_babip,
    ROUND((SUM(batter_counts.Hit)-SUM(batter_counts.Home_Run))/
          (SUM(batter_counts.atBat)-SUM(batter_counts.Strikeout)-SUM(batter_counts.Home_Run)+SUM(batter_counts.Sac_Fly)),3) AS away_babip
FROM game 
JOIN batter_counts ON game.game_id = batter_counts.game_id AND game.away_team_id = batter_counts.team_id
GROUP BY game.game_id, game.away_team_id;
