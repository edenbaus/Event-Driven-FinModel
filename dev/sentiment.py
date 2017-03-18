### this file is used to add sentiment polarity in existing dataframe
### input: pandas dataframe with date, news title, news content 

import pandas 
from textblob import TextBlob
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')
def senti():
	data=parse()
	Polarity = map(lambda x: TextBlob(x),data.content)
	result=[]
	for i in range(len(Polarity)):
		k = Polarity[i].sentiment.polarity
		result.append(k)
	data['polarity']=result
	df=date_transform(data)
	####################transform data to standard structure###############
	return data

def parse():
	Ticker=rawinput('please input ticker')
	data=open(Ticker+'.csv')
	array=data.read().split('|')
	date=[]
	content=[]
	for i in range(2,len(array)-1):
		if len(array[i])<30:
			date.append(array[i])
		else:
			content.append(array[i])
	dic={}
	dic['date']=date
	dic['content']=content
	df=pd.DataFrame(dic)
	return df

def date_transform(df):
	date=df.date
	y=range(2007,2017)
	year=map(lambda x: str(x), y)

	
	real_date=[]
	for i in range(len(date)):
		if any(word in date[i] for word in year):
		####has year#####
			real_date.append(date[i].split(',')[0]+date[i].split(',')[1])
		else:
			real_date.append(date[i].split(',')[1]+' 2017')
	df.date=real_date
	return df



