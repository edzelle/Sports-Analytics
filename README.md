# Sports-Analytics
Data and analytics about the 2017 baseball season

This project is the beginning of a longer study into the concepts of consistency and streakiness in the game of baseball.
My initial assumption was that baseball players hit hot streaks during the season and that these streaks indicate a better near future performace than long term averages. 

My proceudre is as follows:

1. Collect data:

 I collected daily data samples for 1 day 7 day and 30 day stat totals for all MLB players. I got my data from MLB.com and used the chrome extention Data Miner to create csv files for each of the readings. The files were named with the date they were taken and the length of time the data was collected over. For instance, the file 1_Day_6_19_2017 contains the one day statistics for the hitters on 6/19/2017.

2. Data preprocessing:

First I created a score metric to use as an indicator of player performace and as the target for prediction in the future anaysis. I assigned weights to most of the stat catergories and computed a score that summed up these individual values. These weights can be adjusted to create differnt heuristics for player performance. The weights I used are identical to DraftKings.com's classic scoring rules. The scores are all created using 1 day stats becuase one day performance are what I aim to study.

I then created 3 final data files, one for each of the 1 day, 7 day and 30 day stat totals. A row in each of the files is a player, their 1, 7 or 30 day stat totals and the score of their stats for the next day. For instance, player's entry representing their 7 day total on July 1st has their stat totals for the 7 days leading up to July 1st and the score for their 1 day total on July 2nd. This work is done in the file Data_Preprocessor.py

3. Analysis:

I used Python's sklearn library to interpret the results from the datat I collected. I first created a boxplot of the data to visualize the distribution of scores. In all three cases, the data is very scewed towards 0 with a few outliers with higher scores. Next, I used a multilayer perceptron with 5 hidden layers and 20 units per layer to run a regression on the data. The results indicated that the best predictor of the score was the 7 day stats with a R^2 value of .51. The next best was the 30 stats with a R^2 of .33. The 1 day stats predicted the scores with a R^2 value of .16. All three results indicate that there is a positive correlation between a players previous performance and their next day's score. These results suggest that a player's 7 day performace is a better indicator of his next day score than his 30 day performance. The learning curves that the program creates indicate that there may be a higher precision and recall score if there perception had more units. The data supports the idea that a hot player will continue to perform well. Futher work could be done to see if this trend holds between a players' career stats and their 7 day stats. This is conducted in the file Regression_Analysis.py.
