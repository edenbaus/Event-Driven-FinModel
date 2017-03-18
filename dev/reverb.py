###This file is used to use reverb function on given datasets###
###it creates text file and run in terminal then delete that file####

import os
import pandas as pd
import numpy as np
from nltk import tokenize
import re
def parse(filename):
	txt_file=open(filename,'r')
	txt=open("new.txt",'w')
	info=txt_file.read()
	
	info=txt.read()

	section=info.split('|')
	i=1
	date=[]
	news=[]
	for s in section:
		if i==1:
			date=np.append(date,s)
			print 'date '+s+'\n'
			i=2
		else:
			news_line=s.split('\n')
			j=-2
			while len(news_line[j])==0:
				j=j-1
			line=news_line[j]
			print 'news: '+news_line[j]+'\n'
			news=np.append(news,line)
			i=1
	df={}
	df['date']=date
	df['news']=news
	df=pd.DataFrame(df)
	df.to_csv("../data/db.csv")
	txt.close()

def reverb():
	
	
	filename = raw_input("Please Input filename: ")
	data=pd.read_csv(filename,header=-1)
	news=data.iloc[:,2].dropna()
	reverb=[]
	for each_news in news:
		print each_news
		news_line=filter(lambda x: "BAC" in x, each_news.split())[0]
		text_file = open("single_news.txt", "w")
		text_file.write(news_line)
		text_file.close()
		os.system("java -Xmx512m -jar reverb-latest.jar single_news.txt > output.txt")
		output=open("output.txt",'r')
		info=output.read()
		output.close()
		print info
		info_list=info.split(' ')

		reverb=np.append(reverb,info)
	os.system("rm single_news.txt")
	os.system("rm output.txt")
	data['reverb']=reverb
	data.to_csv("afterprocess.csv")

parse("../data/C.txt")




