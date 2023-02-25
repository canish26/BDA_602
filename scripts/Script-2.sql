use baseball;
-- batter_counts table
SELECT batter, game_id, Hit, atBat FROM batter_counts ORDER BY batter, game_id LIMIT 0, 20;

-- Game Table
SELECT game_id, local_date, YEAR(local_date) AS game_year FROM game LIMIT 0, 20;
-- Creating Tables to store the calculated averages
-- Historic batting average for each player
DROP TABLE IF EXISTS batt_avg_hist;

CREATE TABLE IF NOT EXISTS batt_avg_hist as
SELECT 
    batter, 
    SUM(hit) AS total_hits, 
    SUM(atBat) AS total_at_bats, 
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM 
    batter_counts bc 
GROUP BY 
    batter ;
#ALTER TABLE batt_avg_hist ADD PRIMARY KEY (batter);
    
SELECT * FROM batt_avg_hist ;

-- Annual batting average for each player

DROP TABLE IF EXISTS batt_avg_annual;

CREATE TABLE IF NOT EXISTS batt_avg_annual as
SELECT batter AS Batter,
      YEAR(game.local_date) AS For_Year,
       SUM(hit) AS total_hits, 
    SUM(atBat) AS total_at_bats, 
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM batter_counts AS bc
INNER JOIN game ON bc.game_id = game.game_id
GROUP BY
        Batter, For_Year
ORDER BY 
        Batter, For_Year;
       
ALTER TABLE batt_avg_annual ADD PRIMARY KEY (batter, For_Year);

SELECT * FROM batt_avg_annual ;

#############
CREATE TABLE IF NOT EXISTS game_bat_int as
select bc.game_id, bc.batter, local_date, bc.hit, bc.atBat, g.local_date
from batter_counts bc
inner join game g on bc.game_id=g.game_id;

SELECT * from game_bat_int limit 0,20;

ALTER table game_bat_int add index batter_index(batter);

ALTER table game_bat_int add index date_index(local_date);


##############
DROP TABLE temp_roll_avg_intermediate;
-- create temp table to store data
CREATE TABLE temp_roll_avg_intermediate
AS (
  SELECT bc.batter, g.local_date, bc.hit, bc.atBat
  FROM batter_counts AS bc
  LEFT JOIN game AS g
  ON bc.game_id = g.game_id
);
-- creating index
CREATE INDEX idx_batter ON temp_roll_avg_intermediate(batter);
CREATE INDEX idx_date ON temp_roll_avg_intermediate(local_date);
CREATE INDEX idx_batter_date ON temp_roll_avg_intermediate(batter, local_date);

DROP table tmp_rolling_avg;
-- Rolling avg
CREATE TEMPORARY TABLE temp_roll_avg
AS (
  SELECT a.batter, a.local_date, (SUM(b.hit) / NULLIF(SUM(b.atBat), 0)) AS rolling_avg
  FROM temp_roll_avg_intermediate AS a
  JOIN temp_roll_avg_intermediate AS b
  ON a.batter = b.batter AND a.local_date > b.local_date AND b.local_date BETWEEN a.local_date - INTERVAL 100 DAY AND a.local_date
  WHERE a.batter = 435623
  GROUP BY a.batter, a.local_date
);

SELECT * FROM temp_roll_avg
ORDER BY local_date DESC;
