import glob
import pandas as pd
import numpy as np
import os.path

def score_row(df, w, columns):
	runs = df[[8]]	
	doubles = df[[10]]
	triples = df[[11]]
	homeruns = df[[12]]
	singles = df[[9]] - (doubles + triples + homeruns)
	rbis = df[[13]]
	walks = df[[14]]
	stolenbases = df[[16]]
	hbp = df[[23]]
	row_scores = w[0]*runs + w[1]*singles + w[2]*doubles + w[3]*triples + w[4]*homeruns + w[5]*rbis + w[6]*walks + w[7]*hbp + w[8]*stolenbases
	print(row_scores)
	return row_scores

path1 = "1_Day_Hitting_Stats\*.csv"
path2 = "7_Day_Hitting_Stats\*.csv"
path3 = "30_Day_Hitting_Stats\*.csv"
flag = True


## This loop iterates through each file in the directories and assign scores to them ##
## Reads in the previous Day's score to memory first so that it can open the file for the next day and add to it ##

## 1. Open previous day file and calculate the score ##
## 2. Save that file to a pandas dataframe in memory ##
## 3. Open the file for the current day and make a df for it ##
## 4. Left join saved df with open file ##
## 5. Save df to masterfile ##
## 6. Repeat for other 2 files of the same date ##
score_df = None

w =[2,3,5,8,10,2,2,2,5] # These values are the weights that are used to compute the score value

metric_columns = [4,8,9,10,11,12,13,14,16,23]


## 1: ## 
idx = 0	
for file in glob.glob(path1):

	
## 2: ##	
	score_df = pd.read_csv(file)
	score_df.fillna(0)
## Creates a new column for the score in the data frame ##
	score_df['Score'] = (w[0]*score_df.iloc[:,8]+w[1]*(score_df.iloc[:,9]+(score_df.iloc[:,10])*-1+ score_df.iloc[:,11]*-1 + score_df.iloc[:,12]*-1)+w[2]*score_df.iloc[:,10]+w[3]*score_df.iloc[:,11]+w[4]*score_df.iloc[:,12]+w[5]*score_df.iloc[:,13]+w[6]*score_df.iloc[:,14]+w[7]*score_df.iloc[:,16]+w[8]*score_df.iloc[:,23])
	
## Casts the ID column to an int in preperation of sorting ##	
	score_df.columns.values[4] = "ID"
	score_df[['ID']] = score_df[[4]].astype(int)

## Minimizes the score dataframe so that we only get the necissary columns in preperation for a join ##
	idxs = np.arange(0,len(score_df.columns),1)
	temp = np.ones(len(score_df.columns), dtype = bool)
	temp[[1,2,4,5,34]] = False

	score_df = score_df.drop(score_df.columns[[idxs[temp]]], axis = 1)

	idx+=1

	score_df = score_df.sort_values('ID')



## 3:##
	df1 = pd.read_csv(glob.glob(path1)[idx])
## 4:##	
	df1.columns.values[4] = "ID"
	df1[[4]] = df1[[4]].astype(int)
	
	df1 = df1.sort_values('ID')

## Joins the score from the previous day with the stats of the current day ##

	X = pd.merge(score_df, df1, how = 'left', on = ['ID', 'Player', 'Pos', 'Team'], left_index = True)
## eliminates rows of NAs and fills remianing NAs with 0s ##
	X = X.drop('RK', axis = 1)
	X = X.dropna(axis = 0, thresh = 10)
	X = X.fillna(0)
	X = X.drop(X.columns[[5]], axis =1)
	X = X.drop(X.columns[[33]], axis =1)
## 5:##

	if(os.path.isfile("1_Day_Data.csv")):
## This opens up the existing file and appends more rows to the end of it ##
		outfile = r"C:\Users\Eli\Desktop\Projects\Baseball\Hitting Stats\1_Day_Data.csv"
		with open(outfile, 'a') as f:
			X.to_csv(f, header = False)

	else:
		X.to_csv("1_Day_Data.csv", sep =",")

## 7 Day dataset __________________________________________________________________________##
## 6:##
	## 3:##
	df2 = pd.read_csv(glob.glob(path2)[idx])
## 4:##	
	df2.columns.values[4] = "ID"
	df2[[4]] = df2[[4]].astype(int)
	
	df2 = df2.sort_values('ID')

## Joins the score from the previous day with the stats of the current day ##

	Y = pd.merge(score_df, df2, how = 'left', on = ['ID', 'Player', 'Pos', 'Team'], left_index = True)
## eliminates rows of NAs and fills remianing NAs with 0s ##
	Y = Y.drop('RK', axis = 1)
	Y = Y.dropna(axis = 0, thresh = 10)
	Y = Y.fillna(0)
	Y = Y.drop(Y.columns[[5]], axis =1)
	Y = Y.drop(Y.columns[[33]], axis =1)
## 5:##

	if(os.path.isfile("7_Day_Data.csv")):
## This opens up the existing file and appends more rows to the end of it ##
		outfile = r"C:\Users\Eli\Desktop\Projects\Baseball\Hitting Stats\7_Day_Data.csv"
		with open(outfile, 'a') as f:
			Y.to_csv(f, header = False)

	else:
		Y.to_csv("7_Day_Data.csv", sep =",")


## 30 Day dataset __________________________________________________________________________##
## 6:##
	## 3:##
	df3 = pd.read_csv(glob.glob(path3)[idx])
## 4:##	
	df3.columns.values[4] = "ID"
	df3[[4]] = df3[[4]].astype(int)
	
	df3 = df3.sort_values('ID')

## Joins the score from the previous day with the stats of the current day ##

	Z = pd.merge(score_df, df3, how = 'left', on = ['ID', 'Player', 'Pos', 'Team'], left_index = True)
## eliminates rows of NAs and fills remianing NAs with 0s ##
	Z = Z.drop('RK', axis = 1)
	Z = Z.dropna(axis = 0, thresh = 10)
	Z = Z.fillna(0)
	Z = Z.drop(Z.columns[[5]], axis =1)
	Z = Z.drop(Z.columns[[33]], axis =1)
## 5:##

	if(os.path.isfile("30_Day_Data.csv")):
## This opens up the existing file and appends more rows to the end of it ##
		outfile = r"C:\Users\Eli\Desktop\Projects\Baseball\Hitting Stats\30_Day_Data.csv"
		with open(outfile, 'a') as f:
			Z.to_csv(f, header = False)

	else:
		Z.to_csv("30_Day_Data.csv", sep =",")