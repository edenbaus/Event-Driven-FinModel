
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv
import random

# Windows users need to specify the path to chrome driver you just downloaded.
# driver = webdriver.Chrome('path\to\where\you\download\the\chromedriver')

def seekalpha():
	driver = webdriver.Chrome()
	url="http://seekingalpha.com/symbol/BAC/news"

	driver.get(url)
	lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
	match=False
	i=1
	while match==False and i<100000:
		i=i+1
		lastCount = lenOfPage
		time.sleep(3)
		lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
		if lastCount==lenOfPage:
			match=True

	csv_file = open('BAC.csv', 'wb')
	writer = csv.writer(csv_file)
	#writer.writerow(['date', 'news'])==> main function
	# Page index used to keep track of where we are.


			


# Find all the reviews.
	newsunit = driver.find_elements_by_xpath('//li[@class="mc_list_li"]')
	for nu in newsunit:
				# Initialize an empty dictionary for each review
		news_dict = {}
				# Use Xpath to locate the title, content, username, date.
		try:
			date = nu.find_element_by_xpath('//span[@class="date"]').text
			news= nu.find_element_by_xpath('//a[@class="market_current_title"]').text
	
			news_dict['date'] = date
			news_dict['news'] = news

			writer.writerow([unicode(s).encode("utf-8") for s in news_dict.values()])
			print "Write to csv file"
			# Locate the next button on the page.

			
			print "wating time is ", 3
			time.sleep(3)
		except Exception as e:
			print e
			break
			csv_file.close()
			driver.close()

seekalpha()
	