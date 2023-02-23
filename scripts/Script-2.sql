use baseball;
SELECT * FROM game LIMIT 0, 20;

create table batt_avg_hist as
SELECT 
    batter, 
    SUM(hit) AS total_hits, 
    SUM(atBat) AS total_at_bats, 
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM 
    batter_counts bc 
GROUP BY 
    batter ;
    
SELECT * FROM batt_avg_hist ;
   
create table batt_avg_annual as
SELECT batter AS Batter,
      YEAR(game.local_date) AS For_Year,
       SUM(hit) AS total_hits, 
    SUM(atBat) AS total_at_bats, 
    CAST(SUM(hit) AS FLOAT) / NULLIF(SUM(atBat), 0) AS batting_average
FROM batter_counts AS bc
INNER JOIN game ON bc.game_id = game.game_id
GROUP BY Batter, For_Year
ORDER BY Batter, For_Year;

SELECT * FROM batt_avg_annual ;

