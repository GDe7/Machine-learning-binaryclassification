# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 07:33:35 2018

@author: Administrator
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot
from matplotlib import pyplot as plt

#------------Features datasets processing-------------
#-------------------Humanobserveddataset concatenation-------------------------
#combine same_pairs and diffn_pairs to one file for next feature extraction
#file1=pd.read_csv(r'C:\Users\Administrator\Desktop\same_pairs.csv')
#file=pd.read_csv(r'C:\Users\Administrator\Desktop\diffn_pairs.csv')
#file2=file.sample(frac=0.003)  #select 900 hundreds datasets
#file4=file1.append(file2)
#formal=file4.sample(frac=1)
#formal.to_csv(r'C:\Users\Administrator\Desktop\formalsamediffn.csv')

#file3=pd.read_csv(r'C:\Users\Administrator\Desktop\HumanObserved-Features-Data.csv')
#samediffnnoB=formal.drop('img_id_B',1,inplace=False)

#Rule:
#filename= filename.drop('column_name', 1)；
#filename.drop('columnname',axis=1, inplace=True)
#filename.drop([filename.columns[[0,1, 3]]], axis=1,inplace=True)   # Note: zero indexed


#I manually replace 'img_id' in Humanobservedfeatures with 'img_id_A',and do feature extraction for A
###Temporarily used:
#mergedataA=pd.merge(formal,file3,on='img_id_A')
###Temporarily used:
#mergedataA.to_csv(r'C:\Users\Administrator\Desktop\mergedataA.csv')
#newmergedataA=pd.read_csv(r'C:\Users\Administrator\Desktop\mergedataA.csv')
#I manually replace 'img_id' in Humanobservedfeatures with 'img_id_B' and do feature combination with feature A
#mergedata=pd.merge(newmergedataA,file3,on='img_id_B',how='left')
#mergedata.to_csv(r'C:\Users\Administrator\Desktop\mergedata.csv',index=None)
#newmergedata=pd.read_csv(r'C:\Users\Administrator\Desktop\mergedata.csv')

#Extract targetvalue from mergedata excel
#targetdata=newmergedata.loc[:,['target']]
#targetdata.to_csv(r'C:\Users\Administrator\Desktop\targetdata.csv',index=None)
#newtargetdata=pd.read_csv(r'C:\Users\Administrator\Desktop\targetdata.csv')

#Usednewmerge=newmergedata.loc[0:,'f1_x':'f9_y']
#Usednewmerge.to_csv(r'C:\Users\Administrator\Desktop\Usednewmerge.csv',index=None)

#Usedtargetdata=newtargetdata.loc[0:]
#Usedtargetdata.to_csv(r'C:\Users\Administrator\Desktop\Usedtargetdata.csv',index=None)

#After doing all of these steps,I manually delete the name of every column and get 
#suitable feature dataset and target dataset called:concatenationfeatures and concatenationtarget
#Both files have totally 1670 rows.


#--------------------------GSC concatenation-------------------------------
#file1=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\same_pairs.csv')
#file2=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\diffn_pairs.csv')
#file3=file1.sample(frac=0.05)
#file4=file2.sample(frac=0.0047)
#file5=file3.append(file4)
#formal=file5.sample(frac=1)
#formal.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\samediffn_pairs.csv',index=None)

#GSCfeatures=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\GSC-Features.csv')

#I manually replace 'img_id' in GSCfeatures with 'img_id_A',and do feature extraction for A
###Temporarily used:
#mergedataA=pd.merge(formal,GSCfeatures,on='img_id_A')
###Temporarily used:
#mergedataA.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\mergedataA.csv')
#newmergedataA=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\mergedataA.csv')
#I manually replace 'img_id' in GSCfeatures with 'img_id_B' and do feature combination with feature A
#mergedata=pd.merge(newmergedataA,GSCfeatures,on='img_id_B',how='left')
#mergedata.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\mergedata.csv',index=None)
#newmergedata=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\mergedata.csv')

#Extract targetvalue from mergedata excel
#targetdata=newmergedata.loc[:,['target']]
#targetdata.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\targetdata.csv',index=None)
#newtargetdata=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\targetdata.csv')

#Usednewmerge=newmergedata.loc[0:,'f1_x':'f512_y']
#Usednewmerge.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\Usednewmerge.csv',index=None)

#Usedtargetdata=newtargetdata.loc[0:]
#Usedtargetdata.to_csv(r'C:\Users\Administrator\Desktop\Usedtargetdata.csv',index=None)

#After doing all of these steps,I manually delete the name of every column and get 
#suitable feature dataset and target dataset called:GSCconcatenationfeatures and GSCconcatenationtargets

#---------------------------Humansubtract------------------------
#file1=pd.read_csv(r'C:\Users\Administrator\Desktop\Usednewmerge.csv')
#file2=file1.loc[:,'f1_x':'f9_x']
#file3=file1.loc[:,'f1_y':'f9_y']
#file2.to_csv(r'C:\Users\Administrator\Desktop\Humansubfile1.csv',index=None)
#file3.to_csv(r'C:\Users\Administrator\Desktop\Humansubfile2.csv',index=None)
fileval2=pd.read_csv(r'C:\Users\Administrator\Desktop\Humansubfile1.csv')
fileval3=pd.read_csv(r'C:\Users\Administrator\Desktop\Humansubfile2.csv')
file4=np.subtract(file2,file3)
file4.to_csv(r'C:\Users\Administrator\Desktop\Humansubtractfeatures.csv')
#Then I mannually delete row indeices and column names and get Humansubfeatures
#---------------------------GSCsubtract-------------------------
#file1=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\Usednewmerge.csv')
#file2=file1.loc[:,'f1_x':'f512_x']
#file3=file1.loc[:,'f1_y':'f512_y']
#file2.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\GSCfile1.csv',index=None)
#file3.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\GSCfile2.csv',index=None)
#fileval2=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\GSCfile1.csv')
#fileval3=pd.read_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\GSCfile2.csv')

#file4=np.subtract(fileval2,fileval3)
#file4.to_csv(r'C:\Users\Administrator\Desktop\GSC-Features-Data\GSCsubtractfeatures.csv',index=None)
#Then I mannually delete row indeices and column names and get GSCsubtractfeatures