###This file is used to use reverb function on given datasets###
###it creates text file and run in terminal then delete that file####

import os
import pandas as pd
import numpy as np
from nltk import tokenize
import re
from sentiment import parse_1

def reverb():
	ticker, df=parse_1()
	news=df.content
	reverb=[]
	company_name= raw_input("please enter company name")
	for each_news in news:
		delimiters=".","?","!","\n\n"
		regexPattern = '|'.join(map(re.escape, delimiters))
		#news_list=re.split(regexPattern,each_news)
		#print len(news_list)
		#news=filter(lambda x: (ticker in x) or (company_name in x), news_list)

	
		news=each_news+'.'
		print news
		f=open("single_news.txt",'w')
		f.write(news)
		f.close()
		os.system("java -Xmx512m -jar reverb-latest.jar single_news.txt > output.txt")
		output=open("output.txt",'r')
		info=output.read()
		output.close()
		print info
		

		reverb=np.append(reverb,info)
	os.system("rm single_news.txt")
	os.system("rm output.txt")
	df['reverb']=reverb
	df.to_csv(ticker+"reverb.csv")
	
reverb()



