#!/bin/bash
set -e

if [[ -z "$(mysql -h localhost -u root -pCsridhar2601@ -e 'SHOW DATABASES LIKE "baseball"')" ]]; then
    echo "Creating baseball database"
    mysql -h localhost -u root -pCsridhar2601@ -e "CREATE DATABASE baseball"
    tar -xzf /data/baseball.sql.tar.gz -C /data/
    mysql -h localhost -u root -pCsridhar2601@ baseball < /data/baseball.sql
else
    echo "baseball database already exists"
fi
# Make sure to replace the path
echo "Done!"

# Calculating the rolling 100 day Batting Average
# Skip intermediary results if they already exist
mysql -h localhost -u root -pCsridhar2601@ -e "USE baseball;

CREATE TABLE IF NOT EXISTS batt_avg_hist AS
SELECT
    batter,
    SUM(hit) AS total_hits,
    SUM(atBat) AS total_at_bats,
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM
    batter_counts bc
GROUP BY
    batter;

SELECT * FROM batt_avg_hist;

CREATE TABLE IF NOT EXISTS batt_avg_annual AS
SELECT
    batter AS Batter,
    YEAR(game.local_date) AS For_Year,
    SUM(hit) AS total_hits,
    SUM(atBat) AS total_at_bats,
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM
    batter_counts AS bc
INNER JOIN game ON bc.game_id = game.game_id
GROUP BY
    Batter, For_Year
ORDER BY
    Batter, For_Year;

ALTER TABLE batt_avg_annual ADD PRIMARY KEY (batter, For_Year);

SELECT * FROM batt_avg_annual;

CREATE TABLE IF NOT EXISTS game_bat_int AS
SELECT
    bc.game_id,
    bc.batter,
    local_date,
    bc.hit,
    bc.atBat,
    g.local_date
FROM
    batter_counts bc
INNER JOIN game g
ON bc.game_id = g.game_id;

SELECT * FROM game_bat_int LIMIT 0,20;

ALTER TABLE game_bat_int ADD INDEX batter_index(batter);

ALTER TABLE game_bat_int ADD INDEX date_index(local_date);

DROP TABLE IF EXISTS temp_roll_avg_intermediate;
CREATE TABLE IF NOT EXISTS temp_roll_avg_intermediate AS (
    SELECT
        bc.batter,
        DATE(g.local_date),
        bc.hit,
        bc.atBat
    FROM
        batter_counts AS bc
    LEFT JOIN game AS g
    ON bc.game_id = g.game_id
    GROUP BY
        batter,
        local_date
);
CREATE INDEX idx_batter ON temp_roll_avg_intermediate(batter);
CREATE INDEX idx_date ON temp_roll_avg_intermediate(local_date);
CREATE INDEX idx_batter_date ON temp_roll_avg_intermediate(batter, local_date);

DROP TABLE IF EXISTS temp_roll_avg;
CREATE TEMPORARY TABLE IF NOT EXISTS temp_roll_avg AS (
    SELECT
        a.batter,
        a.local_date,
        (SUM(b.hit) / NULLIF(SUM(b.atBat), 0)) AS rolling_avg
    FROM
        temp_roll_avg_intermediate AS a
    JOIN
        temp_roll_avg_intermediate AS b
    ON
        a.batter = b.batter
        AND a.local_date > b.local_date
        AND b.local_date BETWEEN a.local_date - INTERVAL 100 DAY AND a.local_date
    GROUP BY
        a.batter,
        a.local_date
);

SELECT * FROM temp_roll_avg ORDER BY local_date DESC;
"

echo "Done!"
