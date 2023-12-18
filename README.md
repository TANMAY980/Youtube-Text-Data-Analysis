# Youtube-Text-Data-Analysis
# how to read csv data or how to load csv file
''' (ETL)first extract the data after  reading  the data  we get raw data and send to the data transformation phase so 
at the end we will get featurized data and we can send this data for data analysis purpose
'''
!pip install seaborn
'''1st read/modify/manipuliating the data using pandas 
  2nd we can perform numerical computation/meadian/variation/percentile value
  3rd data visulaization using matplotlib module or if we need very fast then we can use seaborn
  and for dynamic plot we can use plotly'''
  
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''reading the csv file using pandas with read_csv() 1st parameter is  local path of the file
    1st file path /filename problem is files are named in different in different os / for windows and \ mac so for this
    problem we can solve with the raw string anything between " " is called normal string so we have to convert from normal
    string to raw string just add the r/R so after that whatever the path you entered it doesn't matter either its 
    windows or linux. (UNICODE ERROR issue will resolved )
    below the result we get 2D dataframe so in pandas we have two type of dataframe 1 dimensional 2 dimensional data frame
    1D- array,list (series)   2D -collection of list,array
'''
'''storing the dataframe in dataframe object'''
comments = pd.read_csv(r'E:\template\data analysis/Uscomments.csv',error_bad_lines=False)

'''1st 5 rows in column dataframe'''
comments.head(5)
![Screenshot 2023-12-18 215922](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/5257c7b5-c552-469c-b3ff-42ac247c4add)


'''checking is missing value is present or not 
    isnull() returns the boolean dataframe is false means- not present missing value True- present missing value
'''
comments.isnull()
![Screenshot 2023-12-18 220208](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/fd455f0d-f950-4397-b63e-ba8ec53e8ce3)


'''sum() shows total missing value in features'''
comments.isnull().sum()
![Screenshot 2023-12-18 220259](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/942be263-cf5a-415f-b819-94e8fc356da9)


'''so we have to drop the missing value from the features 
   dropna(inplace=True) remove the missing values
'''
comments.dropna(inplace=True)


'''after dropping the missing value checking again'''
comments.isnull().sum()
![Screenshot 2023-12-18 220344](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/3718c7bd-89a8-4188-ade3-dde93ef3e703)



'''              PERFORMING SENTIMENT ANALYSIS

   performing exploratory data analysis
   analysing the sentiment of user like user1- very helpful sentiment value would lie of between 0 to 1 more close to 1 means
   (positive sentiment) and more closer to 0 means (neutral sentiment)[0,1]
    
   user2-unable to understand sentiment vlaue range will be [-1,0] more closer to -1 it will be more negative sentiment
   more closer to 0 means more neutral sentiment
   so range of the sentiment or polarity value[-1,1]  1=positive sentiment -1 negative sentiment
   
   packages that we use for sentiment analysis is TEXTBLOB
   so install textblob by using pip
'''
!pip install textblob

'''from package import function '''
from textblob import TextBlob
comments.head(6)
![Screenshot 2023-12-18 220430](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/77e4c917-4bb5-4888-9f0a-3ed99c26fbda)

'''wanted to check the textblob'''
'''and wanted to show sentiment of the textblob and if we wanted to see the polarity attribute 0.0 means neutral sentiment'''
TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity
![Screenshot 2023-12-18 220509](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/dd8adba9-000c-40d5-a3ea-e38ab87a7778)

'''checking how many data points present in the csv file
    so we have approx 700k data so it will take time to iterate through this data so we have to take sample data less amount
    of data
    
'''
comments.shape

'''fetching the sample data like from 700k data we want only 1k data or 2k data'''
sample_df=comments[0:1000]
'''1st 1000 sample row'''
sample_df.shape


'''wanted to see the sentiment analysis for each of the row(comment_text) means we have to iterate using for loop
    for comment in comments dataframe with feature ['commnets_text'] and the moment we extract each comment we can pass
    to the textblob and i need polarity value the moment i get the polarity value we have to store each commnets polarity value
    in the list
'''
'''now one scenarios that what if there is an empty comment in the comment then it will return exception thats why we need to
    handle the exceptionusing try exception
'''
polarity=[]
for comment in comments['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)

len(polarity)

'''inserting list in to the dataframe polarity feature
    getting new features in the comments dataframe polarity
'''

comments['polarity']= polarity
comments.head(5)
![Screenshot 2023-12-18 215759](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/9b827906-655f-49bf-aeff-09878a8bf70b)



'''wordcloud analysis of your data
    graphical representation of text data so we are able to see which are my most important key words
    to getting the all the positve feedback 
    dataframe for positive polarity
'''
filter1=comments['polarity']==1
comments_positive=comments[filter1]

'''getting all the negative feedback means getting al the poarity ==-1
    dataframe for negative polarity
'''
filter2=comments['polarity']==-1
comments_negative=comments[filter2]

'''getting  the positive value from  polarity '''
comments_positive.head(5)

![Screenshot 2023-12-18 215618](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/5ebe54e7-4183-4a2c-b7bc-77042d03f2a0)


'''now we need to perform wordcloud analysis'''
!pip install wordcloud

'''from this package we need to import classes like wordCloud STOPWORDS
    so STOPWORD IS THOOSE WORD THAT DOESNT MAKE ANY SENSE IN MY ANALYSIS meaningless words like are,the,and,is,to,from
    so exclude these words

'''
from wordcloud import WordCloud,STOPWORDS

''' unique stopwords define in the wordcloud package'''

set(STOPWORDS)
![Screenshot 2023-12-18 215618](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/4b6335a2-c789-4cae-b0c5-fdab1cde55f4)


'''these are all the rows in the series data'''
comments['comment_text']
![Screenshot 2023-12-18 215426](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/c46d86ad-6d1d-4dec-9e20-44f00268bdbd)


'''checking the datatype
    its a series datastructure
'''
type(comments['comment_text'])


'''so this data is present in the series datastructure for stopwords we need to covert this into string data type
    in python we can do using join()
    join each of the row of comments feature
'''

'''storing all the positive comments which we can find by polarity value we need to convert in a string'''
total_comments_positive= ' '.join(comments_positive['comment_text'])

'''stopwords : set of strings or None
    The words that will be eliminated. If None, the build-in STOPWORDS
    list will be used. Ignored if using generate_from_frequencies.
    these are the list of stopword that we dont need
    after that we need to generate the wordcloud 
    after we getting the string format we need to pass thoose string into wordcloud function 
    '''
wordcloud=WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_positive)


'''
    this for positive comments
if you want to generate  wordcloud as view you can use imshow() from matplotlib
    after that we get the positive keyword that we can show to our stakeholder or client
    this shows positive polarity data
'''
plt.imshow(wordcloud)
plt.axis('off')
![Screenshot 2023-12-18 215345](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/9c3478b8-bedc-4eb9-9aff-b78a980475ab)


'''storing all the negative comments which we can find by polarity value we need to convert in a string'''
total_comments_negative= ' '.join(comments_negative['comment_text'])

wordcloud2=WordCloud(stopwords=set(STOPWORDS)).generate(total_comments_negative)

'''
    this for negative comments
if you want to generate  wordcloud as view you can use imshow() from matplotlib
    after that we get the positive keyword that we can show to our stakeholder or client
    this shows negative polarity data
    these are the keywords that customer are using these comments on the youtube comment section
    
'''
plt.imshow(wordcloud2)
plt.axis('off')
![Screenshot 2023-12-18 215302](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/72e690f9-8e44-4967-8146-6dcfc2013495)


'''PERFORMING EMOJI'S ANALYSIS'''


'''EDA part
    Exploratory Data Analysis (EDA) is a method of analyzing and summarizing datasets to uncover patterns
    trends, and insights. EDA is an important first step in any data analysis
    A key component of EDA is data visualization, which is the graphical representation of data using plots, charts, and graphs
    emojies extensively count means count of the frequent emojies that have been mostly used


'''
!pip install emoji==2.2.0
import emoji
emoji.__version__

'''so we can see in the comment text there is an emoji with the text in the comment text can we extract emoji from the text
    so 1st take the text with the emoji text
    trending 😉

'''
comments['comment_text'].head(6)

![Screenshot 2023-12-18 215210](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/de722980-b805-4f02-acf1-ec006e90d489)

'''extracting the only emoji fro the text'''
comment='trending 😉'

[char for char in comment if char in emoji.EMOJI_DATA]
![Screenshot 2023-12-18 215054](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/c1e711f8-133a-4745-af33-d105de0de47e)


all_emojis_list=[]
for comment in comments['comment_text'].dropna():
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emojis_list.append(char)

all_emojis_list[0:10]

![Screenshot 2023-12-18 215022](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/9fb5a0c1-bfef-4ff4-b50e-19c140bb1865)


'''for each emoji i need count for each of the element we need to check how many time the emoji repeating
    to do this we have to use collection package
'''

from collections import Counter

'''for each emoji i need count for each of the element we need to check how many time the emoji repeating
    to do this we have to use collection package
'''

from collections import Counter

'''finding out the most common emojies count'''
Counter(all_emojis_list).most_common(10)

![Screenshot 2023-12-18 214816](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/076e8656-4f8a-4a1a-b241-1eb37e5325ed)


'''in one list i need emojies and one list we need count of the emoji's
so that we can simply ploted in bar chart'''
Counter(all_emojis_list).most_common(10)[0]
'''with this we are getting the count emoji and the count of the emoji of the 0 th position emoij'''

Counter(all_emojis_list).most_common(10)[0][0]
'''with this indexing we are getting the result of the oth indext o th position value which is emoji'''

Counter(all_emojis_list).most_common(10)[0][1]
'''with this indexing we are getting the result of the oth indext o th position value which is count of the emoji'''
'''so we have to store all the emojis in the list
    so the variable is the [1st]  index and inside this list [0] emoji is present
'''
emojis=[Counter(all_emojis_list).most_common(10)[i][0] for i in range(10)]
emojis
![Screenshot 2023-12-18 214718](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/24c7617b-b8a4-49b3-a84b-b83384d712fa)

frequency=[Counter(all_emojis_list).most_common(10)[i][1] for i in range(10)]
frequency
![Screenshot 2023-12-18 214629](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/6e7a67d5-0547-40e6-b5f9-d5e6b7f51b0a)


'''so we get the both the list 
    1st all the emojis list
    2nd all the frequency of the emojis 
    now we can create bar chart
'''
'''for barchart we need to import plotly and for graph representation graph_objs'''
!pip install plotly


'''for barchart we need to import plotly and for graph representation graph_objs'''
import plotly.graph_objs as go
from plotly.offline import iplot

'''setting the dimension in the bar object gp.bar'''
bar_obj=go.Bar(x=emojis,y=frequency)

'''so if we wanted to this bar_obj in the Barplot we have to use iplot
    we are getting the barchart of the emojis which extreamly used by the user 
'''
iplot([bar_obj])

![Screenshot (383)](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/32fba899-21b6-4435-9938-fdcf56840d86)



'''collecing entire dataset of youtube fro different country
   and doing data transformation  data cleaning data featurization we end up getting prepared data we can do various analysis
   we have to collect the data which is known raw data we have to do data cleaning when clean the data we get featurized data
   now we can do data analysis 
   
   
   whenever you interacting with your system can we create access or modify a file or location since you have to various things
   in your os you have to import os package and using list directory function
'''

import os
files=os.listdir(r'E:\template\data analysis\additional_data')
files
![Screenshot 2023-12-18 214411](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/0ebd85c9-3b43-472e-9ef4-8f79b1fba855)

files_csv=[file for file in files if '.csv' in file]
files_csv

![Screenshot 2023-12-18 214328](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/5e546e33-e3fd-44f8-86e7-b838e927e44b)


'''while collecting the data incase we are any kind of warning so we have to import warnings module'''

import warnings
from warnings import filterwarnings
filterwarnings('ignore')
'''now we have to insert this data into another dataframe 
    full_df="big dataframe" 
    1st define the blank dataframe we will insert each of the dataframe  into this data frame
    can i iterate all the csv file  using for loop 
    for file in file_csv we will create a dataframe of each of the file using pd.read_csv 
    now storing each of the dataframe into the object file current=pd.read()
    now concatinate this dataframe in to the full_df 
    also update full_df=pd.concatinate([destination_Df,current_df])

'''
'''1st define the blank data frame
    adding the path with raw string
    encoding parameter 
    since you have data with various various countries encoding may change depending upon a data country data regional data
    iso-8859-1= best encoding for collection of data with various countries various regions ,error_badline=false
'''
full_df=pd.DataFrame()
path=r'E:\template\data analysis\additional_data'
for file in files_csv:
    current_df=pd.read_csv(path+'/'+file,encoding='iso-8859-1',error_bad_lines=False)
    '''updating the full_df everytime'''
    full_df=pd.concat([full_df,current_df],ignore_index=True)

'''getting rows and colums'''
full_df.shape

![Screenshot 2023-12-18 214231](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/6919b851-0332-4335-81d5-a40ac3c57193)

'''how to export data into csv,json,databases

    we have raw data checking if there duplicate row or irrelivant row is present or not
    if we have then remove it first then load this data
1st checking the duplicate rows present in the df using duplicate function() it returns boolean value True or False 
if present True Or false

'''
'''passing the filter into the full_df and getting the no of duplicate rows present in the df
    duplicate(key=) here key parameter presents the default value is key='first' if 1st 2nd 3rd rows are same if we have a value
    key=first (default) then it says that marks all the rows duplicate except 1st one
    key='last' then 1st and 2nd rows is marked as duplicate
    key='false' then all the rows are marked as duplicate
    if you have to strictly removed all the duplicate rows then key="false"

'''

'''no of duplicates rows'''
full_df[full_df.duplicated()].shape
![Screenshot 2023-12-18 214134](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/60d045ec-7ddf-41ee-ab47-f3179d45a456)


'''removing the duplicate rows'''
full_df=full_df.drop_duplicates()
'''before removing duplicates df 36417 and after removing duplicates df 339525'''
full_df.shape

![Screenshot 2023-12-18 214016](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/46d1f827-393d-4eb5-a701-daf140918112)


'''now after removing the duplicate rows export the data frame into csv,json '''
'''full_df exporting to csv format using to_csv and inside to_csv('enter the path wher you want to export the data/file name')'''

'''exporting the sample data 1000 rows into csv format in the file using path/filename.csv'''
full_df[0:1000].to_csv(r'E:\template\data analysis/youtube_sample.csv',index=False)

''' this is (full_df) large amount of data instead of large data we can use sample data like 1st 10k rows  '''

'''exporting the sample data 1000 rows into csv format in the file using path/filename.json'''
full_df[0:1000].to_json(r'E:\template\data analysis/youtube_sample.json')

'''              NOW IF WE WANT TO EXPORT INTO SOME DATABSE   
    we have to create some engine because using this engine you can connect to your database
    while creating your engine you cn mention what your databse file name
'''

'''creating engine'''
!pip install sqlalchemy
    
from sqlalchemy import create_engine

'''it allow us to connect with the database
    create_engine(url 
    for postgresql
    url"postgresql://
    for sql
    url "sqlite:///path where we want to create database file name"
    create_engine('sqlite:///E:\template\data analysis/youtube_sample.sqlite')
    in order to create a engine for the sqlite database
    then we have to mention sqlite:///database file name
    youtube_sample.sqlite this the databse file which will have deafault table have a users table
    in this users table we can insert our full_df or exporting the dataframe into the table which is inside
    sqlite database  just to do this we have ibuild function in the pandas
    )

'''
engine=create_engine(r'sqlite:///E:\template\data analysis/youtube_sample_data.sqlite')

![Screenshot 2023-12-18 213904](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/c271c9c4-32ed-4a27-b16f-d18dbd03bee4)


'''in this users table we can insert our full_df or exporting the dataframe into the table which is inside
    sqlite database  just to do this we have ibuild function in the pandas
    so to_sql('table',connection,if_exists=just append)
    
    '''
'''WARNING PLEASE DO NOT RUN THIS LINE MORE THAN ONCE BECAUSE WE WILL GET THE VALUE ERROR BY INSERTING THE SAME ROWS AGAIN
    AGAIN
        
'''

'''1000 records succesfully transferd to the sql users table 
    we'll have sqlite file 1000 rows of data into sqlite file which have the table users you can read the data from this 
    file using pandas and sqlite3 packge
'''
full_df[0:1000].to_sql('Users',con=engine ,if_exists='append')
![Screenshot 2023-12-18 213610](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/26038b21-4692-4c1e-b7e6-3ba2faf25e7b)




'''               ANALYZING MOST LIKED CATEGORY
            which category has maximum likes
'''

full_df.head(5)
'''so we  have a feature as category id but do not have the feature as category name we need category name
    like category movies so we need {1:'film & animation'} we can store in the dictionary data structure and map 
    this dictionary to the category_id features once we mapped we will get category name


'''
'''unique category id'''
full_df['category_id'].unique()

![Screenshot 2023-12-18 213549](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/dfdccca4-f789-4947-b980-5480b1bdc402)


'''reading the json file'''
json_df=pd.read_json(r'E:\template\data analysis\additional_data/US_category_id.json')

'''items holds the dictionary itself each of the row is a dictionary'''

json_df
'''in json_df items hold dictionary'''

json_df['items'][0]

'''each of the row itself is a dictionaries

  so we will get outer dictionary and in inner dictioray snippet 
  
  {'kind':'youtube category'
  id:1( this is category id  ),{
  snippet:'channel id':'',
  title:'film & animation', for category id 1 title is ;dilm and animation 
  }
  }

 it means for each of the row we will end up getting tilte and category id
'''

'''describing blank dictionary'''
cat_dict={}
'''for item in json_df['items'].values 
    it will return array representation ,i will access each of the item from the items feature which is a dictionary
    itself from this dictionary i need id and title(from each of the row)
    once you access each pair of the each row we can insert the pair into dictionary
    in order to insert the pair(key,value)  into the dictionary dict[key]=value
    so 1st we need title  so item we need to go inside inner dictionary and we have to go key['title']
    item['snippet']['title']=we will get the title so this is the value , i need key so item ['id']
'''
for item in json_df['items'].values:
    cat_dict[int(item['id'])]=item['snippet']['title']
cat_dict
![Screenshot 2023-12-18 213437](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/10138dcb-9b9a-44cd-b5d3-eee58e2a44be)



'''mapping the dictionary  creating new feature into full_df  using the map(cat_dict)'''
full_df['category_name']=full_df['category_id'].map(cat_dict)

full_df.head(4)
![Screenshot 2023-12-18 213317](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/9e936d8c-ced4-4068-98e1-5c689125574d)




'''                        WHICH CATEGORY HAS MAXIMUM LIKES                    '''

''' instead showcasing the maximum likes for each of the category can show beautiful matrix   
    25th percentile each of the category the median value 75 th percentile or maximum value or minimum value so we can using 
                                       BOXPLOT
  '''

'''What is percentile

like we have [0,1,2,3,4........9] so 10th percentile value will be less than specific value
we have set of numbers  25 percentile value is 32 it means that 25 % datapoints are less than 32  
we can say that 75 % datapoint are greaterthan 32  
                            we gonna use WHISKERS 
                            if we want to create boxplot
'''

'''in boxplot we are passing category name  and likes as feature and data frame'''
plt.figure(figsize=(12,8))
sns.boxplot(x='category_name',y='likes',data=full_df)
plt.xticks(rotation='vertical')

![Screenshot 2023-12-18 213149](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/829c1e22-356f-4a81-a2d3-755cb070207b)
![Screenshot 2023-12-18 213124](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/285b3f7b-d359-4073-a768-0d46695b6a3c)


'''we are getting box plot there some categories whic are performing quite well we have couple of data points it shows that
    we have extremely high value of likes some videos

'''

'''     FIND OUT WHETHER AUDIENCE IS ENGAGED OR NOT'''

'''we can think various feature  like what about like rate of the video,dislike rate,comment count rate 
    this will tell directly among all views that how many people like that videos similarly for disklike comment count
    so create these three feature 
'''
'''we have now three new features and previous we had category name features
    now  each of the category we can plot boxplot how feature is performing regarding like rate,comment count,dislike count
'''
full_df.columns
![Screenshot 2023-12-18 212939](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/5455fa64-6eee-4905-b317-bade2c13e8c8)


'''creating boxplot for like rate'''
'''getting the like_rate of the categories'''

plt.figure(figsize=(8,6))
sns.boxplot(x='category_name',y='like_rate',data=full_df)
plt.xticks(rotation='vertical')
plt.show()
![Screenshot 2023-12-18 212847](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/7afe1cf3-2c4f-4bdf-af39-dcbc8abc9a3c)



'''analyzing relationship between views and likes'''
'''regplot jus combination of scatter and regression line plot'''
sns.regplot(x='views',y='likes',data=full_df)
![Screenshot 2023-12-18 212716](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/65126582-e787-4c65-8771-9974929249e6)



'''we are getting there is a straigh line means if views increase like will always increase in a same way in a linear fashion 
    end up getting straight line if you want to cross check you can  using correlation correlation says that if views increase 1 factor
    or 1 unit  then what unit my like will increase how vies and likes correlated each other
'''
full_df.columns
![Screenshot 2023-12-18 212652](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/4c74810a-9e36-4972-81e9-cd4b7cb126af)


'''accessing the columns '''
'''it will return he dataframe of these three features'''
'''now if we called the correlation function we will get the correlation values

  like vs views it says that it has correlation that ~0.78 it means if views is increase no of 100 likes will increase by 78
'''
full_df[['views','likes','dislikes']].corr()
![Screenshot 2023-12-18 212612](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/1fe0cbf3-1fb2-4ae4-bb52-ce7e7bc3a9ea)


'''we can see the correlation using heatmap   by annot parameter we can show the correlation value '''
sns.heatmap(full_df[['views','likes','dislikes']].corr(),annot=True)

![Screenshot 2023-12-18 212524](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/e72b688e-c306-4c95-b339-4de7b7ffad62)



'''                WHICH CHANNEL HAS THE LARGEST NUMBER OF TENDING VIDEOS               


at the conclusion end of the view you will definitely look on what type of visualization we need so any how if we getting up a 

barchart we can say these are the top channels that have largest no of trending videos

'''

full_df.head(6)
![Screenshot 2023-12-18 212423](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/04a754be-cd2f-45b6-b46e-8ed8759fd150)



'''so for each of the channel title we need to compute total no of rows '''

'''value_counts returns frequency table channel title and count
    we can acheive by group by

'''
full_df['channel_title'].value_counts()
![Screenshot 2023-12-18 204207](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/42f619a9-4622-4d09-bc3c-aba1b8278d34)


'''value_counts returns frequency table channel title and count
    using group by in sorted order using sort_values and reset_index() we will get the dataframe'''
cdf=full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()
cdf
![Screenshot 2023-12-18 204039](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/a258b577-3277-432c-bc23-b8ebbd8a2da8)

'''now i need to manipulate the column name'''
cdf=cdf.rename(columns={0:'total_videos'})
cdf
![Screenshot 2023-12-18 203906](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/d3623f98-eba9-45bf-8b4e-84ec77fcf78a)



'''creating barchart using plotly  
import plotly.express as px using px you can call bar function   px.bar(dataframe) i want to show first 20 dataframe

  px.bar(dataframe name [0:20])

'''



import plotly.express as px
px.bar(data_frame=cdf[0:20],x='channel_title',y='total_videos')

![Screenshot (403)](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/dd77c388-2d33-46c6-9102-d73ca93263b5)


'''we get the visualization 
which of the top 20 channel has most trending videos
 it says that wwe has 663 counts of the channel_videos
'''
'''                    Does punctuatuions in title and tags have any relation with views,likes,dislikes,comments? 


'''

full_df['title']
![Screenshot 2023-12-18 203759](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/39c27f99-d8ab-4abb-92ef-8a4bb884cab2)

full_df['title'][0]
![Screenshot 2023-12-18 203643](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/63a4ca41-ca08-48c5-a762-71a71f970cfc)


'''first we have to count the total no of the punctuation count of the each of the text so that i m able to comeup with a conclusion that
    whether the punctuaion in tilte in text have any relation with likes or not  
    if punctuation count is increase views is increase or not
    if punctuation will decrease whether views will decrease or not
    we need to import string package will help us to compute the total punctuation in any text data
'''
import string
string.punctuation

'''so if we say character in text  Eminem - Walk On Water (Audio) ft. BeyoncÃ©'  
    if each of the char is present in the punctuation then consider the particular character
    getting the len of the punctuation list
'''

len([char for char in full_df['title'][0] if char in string.punctuation])
![Screenshot 2023-12-18 203416](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/8326fae7-555b-444e-95ae-88e288016087)


'''appying each of the title to the pun_count function to find out  len of the punctuation present in the text'''
def punc_count(text):
    return len([char for char in text if char in string.punctuation])

'''now we have to pass the dataframe in this punctuation function in order to find the len of punctuation 
    so we have to use the sample data

'''
sample=full_df[0:10000]

'''creating the new feature in the sample count_punc'''
sample['count_punc']=sample['title'].apply(punc_count)

sample['count_punc']
![Screenshot 2023-12-18 203239](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/b77ff1d4-1b2b-4f80-a991-75dead14240a)



'''now the problem is
 
 Does punctuatuions in title and tags have any relation with views,likes,dislikes,comments?
 
 for this use boxplot
 
 if the punctuation count 0 how are the datapoints distributed ,if the punctuation count 1 how exacty datapoints is 
 distributed  how are views value is distributed

'''

plt.figure(figsize=(8,6))
sns.boxplot(x='count_punc',y='views',data=sample)
plt.show()

'''we can see when punctuation count 2 or 3 the no of views are extremly high'''
![download](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/2c2d1cfc-1cd7-4fd9-a07d-1bb41940e52a)



'''we can do for likes also'''
plt.figure(figsize=(8,6))
sns.boxplot(x='category_name',y='likes',data=sample)
plt.xticks(rotation='vertical')
plt.show()

'''we can see if we have the punctuation count 2,3 there is extremly probability  that you could encounter more no of likes'''
![Screenshot 2023-12-18 202143](https://github.com/TANMAY980/Youtube-Text-Data-Analysis/assets/65010491/3b5d32ca-1ac8-40e5-839e-501adc36dc61)

























































