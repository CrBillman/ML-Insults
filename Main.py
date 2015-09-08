###################################################################################################################
##                                                                                                               ##
##  Main function for classifying comments as insulting or non-insultin to users within an internet forum        ##
##  setting.  This code is meant to solve the Insult Detection Kaggle competition, which can be found at         ##
##  https://www.kaggle.com/c/detecting-insults-in-social-commentary.  Currently, this code contains the          ##
##  following methods for categorization:                                                                        ##
##           X Naive Bayes Rule                                                                                  ##
##                                                                                                               ##
###################################################################################################################

# Fcts is the local module for functions used for insult detection
import Fcts
# Numpy is an external module (That may require loading on HPC/NERSC) that allows C-like matrix manipulation and math.
import numpy as np

Plot_ROC = True
Save_Plots = True
Print_NB_Probs = False


# Initalizes a dictionary, which will be filled based solely on the components of the training set.

Dictionary=[]

# Calls a function to read the data from the training set.  This function is specifically tailored for the format of train.csv.  It returns a numpy matrix
# That is formatted as :
#	[0] : flag for insult, 1 for insult, 0 for non-insult
#	[1] : Year comment was posted
#	[2] : Month comment was posted
#	[3] : Day comment was posted
#	[4] : Hour comment was posted
#	[5] : Minute comment was posted
#	[6] : Second comment was posted
#	[7] : Comment
# If the timestamp is not included, elements 1-6 are filled with "N/A".

Train_Set=Fcts.Read_Data("train.csv")
N_Train=len(Train_Set)

# Creates the dictionary, and counts the number of insults for use in later statistics.  All dictionary elements are lower case, and do not have apostrophes, quotations.

N_Insult=0
for i in xrange(0,N_Train):
	Words=Train_Set[i][7].split()
	for j in xrange(0,len(Words)):
		k=0
		Word_Check=Words[j].strip('"')
		Word_Check=Word_Check.strip('.')
		Word_Check=Word_Check.lower()
		if(Word_Check not in Dictionary):
			Dictionary.append(Word_Check)
	if(Train_Set[i][0]==1):
		N_Insult=N_Insult+1


# Calls a function to generate the probability needed for using the Naive Bayes Rule for insult classification.  NB_Prob is a 3 x (N_words) matrix, which contains three probabilities for each word:
#	[0] : Probability this word is encountered in a comment in the training set
#	[1] : Probability this word is encountered in an insult in the training set
#	[2] : Probability this word is encountered in a non-insult in the training set

NB_Prob=Fcts.Naive_Bayes(Train_Set,Dictionary,N_Insult)

# The NB_Prob matrix is used to generate probabilities for insult classification.  The Bayes rule is more rigorously calculated by summing ln(1-p)-ln(p) for the probability p of each word.  Then,
# this summ is put into a logistic function, 1/(1+e^(Prob_Sum)).  This loop also grades the algorithm by returning the probability we get a correct

Num_Right=0
False_Positive=0
Correct_Insults=0
Insult_Prob=Fcts.Eval_NB(Train_Set,Dictionary,NB_Prob,N_Insult,N_Train)

for i in xrange(0,N_Train):
	if(Train_Set[i][0]==round(Insult_Prob[i])):
		Num_Right=Num_Right+1
	if((Train_Set[i][0]==0) and (Insult_Prob[i]>=0.5)):
		False_Positive=False_Positive+1
	if((Train_Set[i][0]==1) and (Insult_Prob[i]>=0.5)):
		Correct_Insults=Correct_Insults+1

# Prints probability of getting training set classifications correct.
if(Print_NB_Probs):
	print "Probability of getting a correct classification: " + str(Num_Right/float(N_Train))
	print "Probability of generating a false positive: " + str(False_Positive/(False_Positive+float(N_Train-N_Insult)))
	print "Probability of correctly identifying insults: " + str(Correct_Insults/float(N_Insult))

# Plots the ROC, and if Save_Plots==True, it saves it in "ROC.png"
if(Plot_ROC):
	if(Save_Plots):
		Fcts.ROC(Train_Set,Insult_Prob,"Train-ROC.png",N_Insult)
	else:
		Fcts.ROC(Train_Set,Insult_Prob,None,N_Insult)

Test_Set=Fcts.Read_Data("test_with_solutions.csv")

N_Test=len(Test_Set)
Num_Right=0
False_Positive=0
Correct_Insults=0
N_T_Insult=0

Insult_Prob=Fcts.Eval_NB(Test_Set,Dictionary,NB_Prob,N_Insult,N_Train)

for i in xrange(0,N_Test):
	if(Test_Set[i][0]==round(Insult_Prob[i])):
		Num_Right=Num_Right+1
	if((Test_Set[i][0]==0) and (Insult_Prob[i]>=0.5)):
		False_Positive=False_Positive+1
	if((Train_Set[i][0]==1) and (Insult_Prob[i]>=0.5)):
		Correct_Insults=Correct_Insults+1
	if(Test_Set[i][0]==1):
		N_T_Insult=N_T_Insult+1

if(Plot_ROC):
	if(Save_Plots):
		Fcts.ROC(Test_Set,Insult_Prob,"Test-ROC.png",N_T_Insult)
	else:
		Fcts.ROC(Test_Set,Insult_Prob,None,N_T_Insult)
