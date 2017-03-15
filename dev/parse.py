###this function is used to parse the txt news#######
import pandas as pd
import numpy as np
def parse(filename):
	data=pd.read_csv(filename,header=-1)
	i=0
	dic={}
	title=[]
	content=[]
	time=[]
	comment=[]
	j=0
	while j<data.shape[0]:
		if i==0:
			title=np.append(title,data.iloc[j,0])
			i=i+1
		elif i==2:
			buff=data.iloc[j,0].split('|')
			if len(buff)==2:
				time=np.append(time,buff[0])
				comment=np.append(comment,buff[1])
			else:
				time=np.append(time,buff[0])
				comment=np.append(comment,'0 Comment')
			i=0
		else:
			content=np.append(content,data.iloc[j,0])
	dic['title']=title
	dic['content']=content
	dic['time']=time
	dic['comment']=comment
	df=pd.DataFrame(dic)
	df.to_csv('BAC_afterprocess.csv')

parse('../data/BAC.csv')

			




