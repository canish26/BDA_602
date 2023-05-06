#!/bin/bash

sleep 30

if [[ -z "$(mysql -h maria_db -u root -pCsridhar2601@ -e 'SHOW DATABASES LIKE "baseball"')" ]]; then
    echo "Creating baseball database"
    mysql -h maria_db -u root -pCsridhar2601@ -e "CREATE DATABASE baseball"
    mysql -h maria_db -u root -pCsridhar2601@ baseball < baseball.sql
else
    echo "baseball database already exists"
fi

echo "Done!"

# Calculating the rolling 100 day Batting Average
# Skip intermediary results if they already exist
mysql -h maria_db -u root -pCsridhar2601@ -e "USE baseball;

CREATE TABLE IF NOT EXISTS batt_avg_hist AS
SELECT
    batter,
    SUM(hit) AS total_hits,
    SUM(atBat) AS total_at_bats,
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM
    batter_counts
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
    batter_counts
INNER JOIN game ON batter_counts.game_id = game.game_id
GROUP BY
    Batter, For_Year
ORDER BY
    Batter, For_Year;

ALTER TABLE batt_avg_annual ADD PRIMARY KEY (Batter, For_Year);

SELECT * FROM batt_avg_annual;

DROP TABLE IF EXISTS temp_roll_avg_intermediate;
CREATE TABLE IF NOT EXISTS temp_roll_avg_intermediate AS (
    SELECT
        batter_counts.batter,
        DATE(game.local_date),
        batter_counts.hit,
        batter_counts.atBat
    FROM
        batter_counts
    LEFT JOIN game
    ON batter_counts.game_id = game.game_id
    GROUP BY
        batter_counts.batter,
        DATE(game.local_date),
        batter_counts.hit,
        batter_counts.atBat
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

SELECT DISTINCT bc.batter,
(SELECT CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0)
FROM batter_counts bc2
WHERE bc2.batter = bc.batter AND bc2.game_id = '12560') AS batting_average
FROM batter_counts bc
WHERE bc.game_id = '12560';"> output_file.csv
echo "Done!"
