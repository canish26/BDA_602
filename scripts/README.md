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

BFP: total batters faced in historical games(number of plate appearances for an offense is 3×(Innings) + (Runs scored) + (Runners left on base).)

IP: total innings pitched in historical games(pitcher's total number of pitches thrown/total number of innings pitched)

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

### Some of the Good Performing Features that are observed

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
Feature engineering involves transforming raw data into features that are more meaningful for a machine learning model to use in making predictions. In the case of baseball statistics, we can engineer features by breaking down the raw data into continuous and categorical features.

### Continuous Features:

Batting Average: Batting average is a continuous feature that measures a player's success in hitting the ball. It can be used as a feature to evaluate a player's offensive performance.
On-Base Percentage: On-base percentage (OBP) is a continuous feature that measures a player's ability to reach base, whether through a hit or a walk. It is a more advanced metric than batting average and is commonly used in sabermetrics.
Slugging Percentage: Slugging percentage (SLG) is a continuous feature that measures a player's power, or ability to hit for extra bases. It is calculated by dividing the total number of bases a player has by their total number of at-bats.

### Categorical Features:

Handedness: A player's handedness, whether they are left-handed or right-handed, can be used as a categorical feature. Some players may perform better against pitchers of a certain handedness, so this feature could be used to help predict performance against different types of pitchers.
Position: A player's position on the field, such as pitcher, catcher, or outfielder, is another categorical feature that could be used to predict performance. Certain positions may require different skill sets, so this feature could be used to evaluate how well a player performs at their position.
Team: The team a player is on is another categorical feature that could be used to predict performance. Different teams may have different playing styles or strategies, which could affect a player's performance.
Other features that could be engineered from baseball statistics include the player's age, experience, and historical performance, as well as the park factors of the stadium they are playing in. These features could be used to build more complex machine learning models to predict player performance or team success.


Most columns, except game id, game date, and HTWins, had some null values. 
This is because when we use data from games before a certain date to calculate statistics until that game, 
we may encounter missing data for a team's first match in our dataset, resulting in null values.

In addition, when performing mathematical operations on rows with null values, the result will also be null.
This is particularly true for starting pitcher features since many starting pitchers may not have any data before their first game in the available data.

After feature generation, feature engineering was done in three steps.
The resulting data, which had been handled to generate predictor reports.

### P and T value Scores

Which resulted in generating the p values, t scores and random forest variable importance for the continuous predictor types.
Weighted and Unweighted mean of responses for both categorical and continuous predictors with their respective pattern plots and mean of response plots.

In the context of a baseball dataset, p-values can be used to determine the significance of the relationships between the features and the outcome variable, such as the outcome of a baseball game. For example, if a feature has a low p-value, it suggests that it is statistically significant and has a strong relationship with the outcome variable.

The t-value is a measure of the difference between the means of two groups, relative to the variance within the groups. It is used to determine whether the difference between the means is statistically significant, and is calculated by dividing the difference between the means by the standard error of the difference.

In the context of a baseball dataset, t-values can be used to compare the means of different groups or to test the significance of the differences between the means. For example, the t-value can be used to compare the mean performance of teams with different win-loss records or to test the significance of the difference in performance between teams in different divisions or leagues.


### Correlation and Bruteforce

Correlation and Brute Force are two methods that can be used for feature selection and to identify the most important features for predicting the outcome of a game.

The number of runs scored by a team may be strongly correlated with their likelihood of winning a game. Correlation analysis can help to identify important features and to remove redundant or irrelevant features that do not have a strong relationship with the outcome variable.

Brute Force method feature selection involves testing all possible combinations of features to identify the best set of features for predicting the outcome variable. This method can be computationally intensive and time-consuming, but can potentially identify the most important features and achieve high prediction accuracy. 

## MODELS:
These models were applied to a baseball dataset, which had undergone feature engineering and handling of missing data. The goal of the models was to predict the outcome of a baseball game, based on the available features.

Logistic Regression: This is a classification model that uses a logistic function to model the probability of a binary outcome. In this case, it was used to predict the outcome of a baseball game, with an accuracy score of 0.517.

Random Forest Classifier: This is an ensemble learning method that builds a multitude of decision trees and combines their results to make predictions. It has an accuracy score of 0.529 in predicting the outcome of a baseball game.

Decision Tree: This is a classification model that uses a tree-like model of decisions and their possible consequences. It has an accuracy score of 0.502 in predicting the outcome of a baseball game.

AdaBoost: This is a boosting ensemble method that combines multiple weak classifiers to create a strong classifier. It has an accuracy score of 0.378 in predicting the outcome of a baseball game.

XGBoost: This is a gradient boosting framework that uses decision trees as base learners. It has an accuracy score of 0.422 in predicting the outcome of a baseball game.

Overall, the Random Forest Classifier had the highest accuracy score among the models tested, indicating that it was the most effective model for predicting the outcome of a baseball game using the given features. However, it is important to note that the accuracy of the models may vary depending on the quality and quantity of the features used, as well as the size and diversity of the dataset.


## RESULTS AND CONCLUSION
![WHIP PLOT](https://github.com/canish26/BDA_602/blob/final/scripts/mean_WHIP.png)
![ROC Curve](https://github.com/canish26/BDA_602/blob/final/scripts/ROC.png)
