import unicodedata
import unidecode
import string
import numpy as np
import matplotlib.pyplot as plt

# Reads in the data.  Must be in the format of train.csv -- Flag,Timestamp,Comment.  Separates the Timestamp into different elements of the list.  If the timestamp is absent, 
# the elements 1-6 are filled with "N/A"

def Read_Data(filename):
	IN=open(filename,"r")
	IN.readline()
	Data_Set=[]
	while True:
		line=IN.readline()
		if(not line):
			break
		line_list=list(line)
		Flag=int(line_list[0])
		for i in xrange(2,len(line_list)):
			if(line_list[i]==","):
				if(i==2):
					timestamp=None
				else:
					timestamp=line_list[2:(i-1)]
				Comment="".join(line_list[(i+1):len(line_list)])
				break
		if(timestamp):
			Year="".join(timestamp[0:4])
			Month="".join(timestamp[4:6])
			Day="".join(timestamp[6:8])
			Hour="".join(timestamp[8:10])
			Minute="".join(timestamp[10:12])
			Second="".join(timestamp[12:14])
		else:
			Year="N/A"
			Month="N/A"
			Day="N/A"
			Hour="N/A"
			Minute="N/A"
			Second="N/A"
		
		Data_Row=[Flag,Year,Month,Day,Hour,Minute,Second,Comment]
		Data_Set.append(Data_Row)
	return Data_Set

#Not currently used -- not working.
def Unicode_to_ASCII(in_string):
	#table = {
	#	' ' : '\\xa',
	#	'\n' : None
	#	}
	in_string=unicode(in_string)
	out_string=in_string.encode("utf-8")
	out_string=in_string.encode('ascii',errors='backslashreplace')
	#tbl = string.maketrans('\\xa',' ')
	#out_string=string.translate(tbl)
	return out_string

# Calculates the probabilities of encountering each word in a comment, insult, and non-insult.  Returns them in the matrix Prob.

def Naive_Bayes(Train,Dictionary,Class_Count):
	N_Words=len(Dictionary)
	N_Train=len(Train)
	Count=np.zeros([N_Words,3])
	for i in xrange(0,N_Train):
		Words=Train[i][7].split()
		T_Count=np.zeros([N_Words,3])
		for j in xrange(0,len(Words)):
			Word_Check=Words[j].strip('"')
			Word_Check=Word_Check.strip('.')
			Word_Check=Word_Check.lower()
			if(Word_Check in Dictionary):
				Dict_Index=Dictionary.index(Word_Check)
				T_Count[Dict_Index,0]=1
				if(Train[i][0]==1):
					T_Count[Dict_Index,1]=1
				else:
					T_Count[Dict_Index,2]=1
		Count=Count+T_Count

	Prob=np.matrix([Count[:,0]/N_Train,Count[:,1]/Class_Count,Count[:,2]/(N_Train-Class_Count)])
	return Prob

def Eval_NB(Eval_Set,Dictionary,NB_Prob,N_Insult,N_Train):
	N_Eval=len(Eval_Set)
	Insult_Prob=np.empty([N_Eval])
	for i in xrange(0,N_Eval):
        	Words=Eval_Set[i][7].split()
        	Prob_Sum=0.0
        	Word_Count=0
        	for j in xrange(0,len(Words)):
                	Word_Check=Words[j].strip('"')
                	Word_Check=Word_Check.strip('.')
                	Word_Check=Word_Check.lower()
                	if(Word_Check in Dictionary):
                        	Word_Index=Dictionary.index(Word_Check)
                        	Ind_Prob=NB_Prob[1,Word_Index]*(N_Insult/float(N_Train))/(NB_Prob[1,Word_Index]*(N_Insult/float(N_Train))+NB_Prob[2,Word_Index]*(1-N_Insult/float(N_Train)))
                        	if(Ind_Prob==1.0):
                        	        Ind_Prob=1.0-1E-10
                        	elif(Ind_Prob==0.0):
                                	Ind_Prob=1E-10
                        	Prob_Sum=Prob_Sum+(np.log(1.0-Ind_Prob)-np.log(Ind_Prob))
                        	Word_Count=Word_Count+1
        	if(Word_Count==0):
                	Insult_Prob[i]=0.5
        	else:
                	if((np.isneginf(Prob_Sum))  or Prob_Sum<=-70):
                        	Insult_Prob[i]=1.0
                	elif((np.isinf(Prob_Sum)) or Prob_Sum>=70):
                        	Insult_Prob[i]=0.0
                	else:
                        	Insult_Prob[i]=1/(1+np.exp(Prob_Sum))
	return Insult_Prob

# Calculates the ROC for different cut-offs and then plots it.  If Save==True, it saves it in "ROC.png", otherwise it displays it.

def ROC(Train,Prob,Save,N_Insult):
	N_Train=len(Train)
	Bin_Width=0.01
	Bins=np.arange(start=0.0,stop=1.0+Bin_Width,step=Bin_Width)
	N_Bins=len(Bins)
	False_Positives=np.empty([N_Bins])
	True_Positives=np.empty([N_Bins])
	for i in xrange(0,N_Bins):
		TP_Count=0
		FP_Count=0
		for j in xrange(0,N_Train):
			if(Prob[j]>=Bins[i]):
				if(Train[j][0]==1):
					TP_Count=TP_Count+1
				else:
					FP_Count=FP_Count+1
		True_Positives[i]=TP_Count/float(N_Insult)
		False_Positives[i]=FP_Count/float(FP_Count+N_Train-N_Insult)
	plt.plot(False_Positives,True_Positives)
	plt.plot(Bins,Bins,'k--')
	plt.xlim([0.0,1.0])
	plt.xlabel("False Positive Rate")
	plt.ylim([0.0,1.0])
	plt.ylabel("True Positive Rate")
	if(Save):
		plt.savefig(Save)
	else:
		plt.show()
	print (max(False_Positives)-np.trapz(True_Positives,False_Positives))
