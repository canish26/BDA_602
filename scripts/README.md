## INTRODUCTION

![Baseball Image](https://www.shutterstock.com/image-photo/baseball-player-throws-ball-on-260nw-1131760214.jpg)


### <a href="https://en.wikipedia.org/wiki/Baseball_statistics">Baseball</a>


The project aims to predict the likelihood of HomeTeam winning in baseball games,
providing valuable insights to fans, bettors, and analysts. Leveraging statistical models and historical data,
our predictions empower individuals to make informed decisions when it comes to betting on games or gaining a better understanding of the potential outcome.



Baseball is a sport that has been played for over 150 years, and over that time, it has accumulated a vast amount of statistics and features that have become integral to the game. Here is a brief history of baseball statistics and features:

Batting Average: One of the oldest and most well-known statistics in baseball is batting average, which measures a player's success in hitting the ball. It is calculated by dividing the number of hits by the number of at-bats. Batting average was first used in the 19th century and is still used today to evaluate a player's offensive performance.

Pitching Statistics: In addition to batting average, baseball also has several pitching statistics that are used to evaluate a pitcher's performance. Earned run average (ERA) measures the average number of runs a pitcher allows per nine innings pitched. Strikeouts (K) and walks (BB) are also commonly used to evaluate a pitcher's effectiveness.

Home Runs: Home runs are one of the most exciting aspects of baseball, and they have been a part of the game since its early days. Babe Ruth is perhaps the most famous home run hitter in baseball history, and his record of 714 home runs stood for many years until it was broken by Barry Bonds in 2007.

baseball is a sport that has a rich history of statistics and features that have evolved over time. From the earliest days of batting average to the more advanced sabermetrics of today, these statistics and features are an essential part of the game and continue to be used to evaluate player and team performance.

This project is build using Python, MariaDB. With the appropriate dependencies will provide consistent results, ensuring that the analysis can be easily
replicated by others. This not only enables others to verify the findings, but also fosters collaboration and
allows for further improvement and refinement of the project. The use of Docker also ensures that any environment
dependencies are met, minimizing the possibility of errors or discrepancies that may arise from differences in
operating systems or packages. Overall, prioritizing reproducibility enhances the credibility and reliability of
the project.
## DATASET:
The Baseball Dataset used in this project is <a href="https://teaching.mrsharky.com/data/baseball.sql.tar.gz" target="_blank">Baseball Dataset</a>.

It includes detailed information about each game, which has been grouped into different tables for convenience.
Tables such as boxscore, game, pitcher_counts, and team_batting_counts have been used for this project.

However, like any dataset, this one also has its issues. For example, the target variable HTWins, which indicates whether the HomeTeam won or not, was generated using the boxscore.home_runs and boxscore.away_runs columns as the boxscore.winner_home_or_away column had incorrect data. The columns related to Caught Stealings and Stolen Bases in all tables are all zeroes, and combining duplicate columns like forceout & force_out, flyout & fly_out, and lineout & line_out would be necessary to use them in any feature.

Moreover, the data in the dataset has other errors, which will be highlighted in the subsequent sections. The aim of this project is to deal with these issues and create a reliable statistical model for predicting whether HomeTeam wins for a baseball game.
## TERMINOLOGY:
These are some of the features used in this project 
BFP: total batters faced in historical games
IP: total innings pitched in historical games
BB9_HI: walks per nine innings pitched in historical games
HA9: hits allowed per nine innings pitched in historical games
HRA9: home runs allowed per nine innings pitched in historical games
SO9: strikeouts per nine innings pitched in historical games
IP: total innings pitched in recent games
BB9: walks per nine innings pitched in recent games
WHIP: walks plus hits per innings pitched in recent games
BFP_ROLL: total batters faced in recent games
SOPercent: strikeout percentage in historical games
WHIP_HIST: walks plus hits per innings pitched in historical games
CERA_HIST: earned run average in historical games
CERA_ROLL: earned run average in recent games

Some of the Good Performing Features that are observed

TEAM PITCH STATS
WHIP 
Hits per 9 Innings Pitched 
Strikeout to walk 
Strikeouts per 9 Innings Pitched 

Some of the GARBAGE Stats performimng features that were observed

No of Innings Finished by Starting Pitcher
At Bats per Pitches thrown 
Strikeouts per 9 Innings Pitched 
Team Batting Stat

SOME OBSERVATIONS

Rolling Averages, especially when the rolling window is not significantly large compared to the data's underlying frequency.Therefore, it is not surprising to see that features generated with 100 day rolling averages are highly correlated with each other.

home teams (HT) and away teams (AT), it is possible that the features may capture different aspects of the game performance for each team. Taking the difference of the features can help to highlight these differences. However, we may lose some data  when taking the difference of correlated features.



## FEATURE ENGINEERING:

## MODELS:
Models worked on are:
Logistic Regression-0.547
Random Forest Classifier-0.529
Decision tree-0.502
AdaBoost-
XGBoost-

## RESULTS AND CONCLUSION
