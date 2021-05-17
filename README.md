# Cryptocurrency Data Science Tutorial

By Owen Knott and Jake Baldwin

## Data Collection and Curation
  When beginning a data analysis on anything, the first step is to find some data to work with. Often, data is not available for us to use in perfect form. This means that we must either find a way to get the data from the internet or clean the data to make it useable for ourselves. In our case, there is a database recording valueable information on crypto currencies online. We can simply download this database and use the information for ourselves. The database we used is located at: https://github.com/oknott14/data_host. The data is currently stored in a repository with each coin containing a seperate .csv file. Since seperated data is hard to work with, we must first compile these files into one large table. We will be using sqlite3 and the pandas python library for this project, but any database library will work.

  First, lets import all of our necessary libraries, and prepare the pytrends library for scraping Google Trends data


```python
import pandas as pd
import sqlite3
from os import listdir
import requests as rq
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.pipeline import Pipeline
import datetime as dt
try:
  from pytrends.request import TrendReq
except:
  !pip install pytrends
  from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-us', tz=360)
```

    Collecting pytrends
      Downloading https://files.pythonhosted.org/packages/96/53/a4a74c33bfdbe1740183e00769377352072e64182913562daf9f5e4f1938/pytrends-4.7.3-py3-none-any.whl
    Requirement already satisfied: pandas>=0.25 in /usr/local/lib/python3.7/dist-packages (from pytrends) (1.1.5)
    Requirement already satisfied: requests in /usr/local/lib/python3.7/dist-packages (from pytrends) (2.23.0)
    Requirement already satisfied: lxml in /usr/local/lib/python3.7/dist-packages (from pytrends) (4.2.6)
    Requirement already satisfied: numpy>=1.15.4 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25->pytrends) (1.19.5)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25->pytrends) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.25->pytrends) (2018.9)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests->pytrends) (2020.12.5)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests->pytrends) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests->pytrends) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests->pytrends) (3.0.4)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.7/dist-packages (from python-dateutil>=2.7.3->pandas>=0.25->pytrends) (1.15.0)
    Installing collected packages: pytrends
    Successfully installed pytrends-4.7.3


Next, lets create our master table conntaining all of our crypto-currencies. For later use, we will format the date as a python datetime object and add a date key column.


```python
#Clone directory and read files
path = '/content/data_host'
try:
  files = listdir(path)
except:
  !git clone https://github.com/oknott14/data_host 
  files = listdir(path)
files.remove('.git')
files.sort() 
sqlName = files.pop() #get sql db file name
if sqlName == 'crypto_data.sql-journal':
  sqlName = files.pop()
#Connect to sql database
conn = sqlite3.connect(f"{path}/{sqlName}")

#Empty Database for tutorial if there are already tables
cur = conn.cursor()
tables = [x[0] for x in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
for table in tables:
  cur.execute(f"DROP TABLE {table}")
cur.close()

#Read Tables into sql
for file in files:
  tbl = pd.read_csv(f"{path}/{file}")
  tbl['DateKey'] = tbl['Date'].apply(lambda c: int(c[0:10].replace('-','')))
  cols = list(tbl.columns)
  cols.remove('SNo')
  tbl[cols].to_sql("Master",conn,if_exists='append',index=False)

frame = pd.read_sql("SELECT * FROM Master",conn)
frame.head()
```

    Cloning into 'data_host'...
    remote: Enumerating objects: 28, done.[K
    remote: Counting objects: 100% (28/28), done.[K
    remote: Compressing objects: 100% (27/27), done.[K
    remote: Total 28 (delta 1), reused 24 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (28/28), done.





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
      <th>DateKey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-05 23:59:59</td>
      <td>55.112358</td>
      <td>49.787900</td>
      <td>52.675035</td>
      <td>53.219243</td>
      <td>0.000000e+00</td>
      <td>8.912813e+07</td>
      <td>20201005</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-06 23:59:59</td>
      <td>53.402270</td>
      <td>40.734578</td>
      <td>53.291969</td>
      <td>42.401599</td>
      <td>5.830915e+05</td>
      <td>7.101144e+07</td>
      <td>20201006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-07 23:59:59</td>
      <td>42.408314</td>
      <td>35.970690</td>
      <td>42.399947</td>
      <td>40.083976</td>
      <td>6.828342e+05</td>
      <td>6.713004e+07</td>
      <td>20201007</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-08 23:59:59</td>
      <td>44.902511</td>
      <td>36.696057</td>
      <td>39.885262</td>
      <td>43.764463</td>
      <td>1.658817e+06</td>
      <td>2.202651e+08</td>
      <td>20201008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-09 23:59:59</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>8.155377e+05</td>
      <td>2.356322e+08</td>
      <td>20201009</td>
    </tr>
  </tbody>
</table>
</div>



Now, lets scrape our data using the unique names from our SQL Database, and query Google Trends with these names. By grabbing the unique names and merging the dataframes we can create one dataframe which is almost ready to merge with our SQL dataframe. We will modify the data that pytrends gives us by creating a startDate and endDate column along with start and end date key columns. The two key columns will make it easier to locate dates within the week of search results.


```python
query_names = frame['Name'].unique() # Shows us the unique Cryptocurrencies we have in the original dataset

trends_df = None

for i in range(0, len(query_names), 5):
  pytrends.build_payload(query_names[i: i + 5]) # Queries google trends a subsection of query_names, to later merge
  try:
    trends_df = trends_df.merge(pytrends.interest_over_time().drop(columns='isPartial'), left_index=True, right_index=True)
  except:
    trends_df = pytrends.interest_over_time().drop(columns=['isPartial'])
    pass

#Create start and end date columns with matching key columns
trends_df['endDate'] = trends_df.index
idt = trends_df.iloc[0,23].date()
trends_df['startDate'] = [pd.Timestamp(idt.year, idt.month, idt.day - 7)] + trends_df['endDate'].tolist()[:-1]
for val in ['end','start']:
  trends_df[f'{val}DateKey'] = trends_df[f'{val}Date'].apply(lambda c: int(c.date().strftime('%Y%m%d')))
trends_df.index = range(0,len(trends_df))
trends_df.to_sql("Trends",conn)
trends_df.head()
```

    /usr/local/lib/python3.7/dist-packages/pandas/core/generic.py:2615: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      method=method,





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Aave</th>
      <th>Binance Coin</th>
      <th>Bitcoin</th>
      <th>Cardano</th>
      <th>Chainlink</th>
      <th>Cosmos</th>
      <th>Crypto.com Coin</th>
      <th>Dogecoin</th>
      <th>EOS</th>
      <th>Ethereum</th>
      <th>IOTA</th>
      <th>Litecoin</th>
      <th>Monero</th>
      <th>NEM</th>
      <th>Polkadot</th>
      <th>Solana</th>
      <th>Stellar</th>
      <th>Tether</th>
      <th>TRON</th>
      <th>USD Coin</th>
      <th>Uniswap</th>
      <th>Wrapped Bitcoin</th>
      <th>XRP</th>
      <th>endDate</th>
      <th>startDate</th>
      <th>endDateKey</th>
      <th>startDateKey</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-05-22</td>
      <td>2016-05-15</td>
      <td>20160522</td>
      <td>20160515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>1</td>
      <td>4</td>
      <td>7</td>
      <td>3</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-05-29</td>
      <td>2016-05-22</td>
      <td>20160529</td>
      <td>20160522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>21</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-06-05</td>
      <td>2016-05-29</td>
      <td>20160605</td>
      <td>20160529</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>21</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>3</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-06-12</td>
      <td>2016-06-05</td>
      <td>20160612</td>
      <td>20160605</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>4</td>
      <td>6</td>
      <td>2</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2016-06-19</td>
      <td>2016-06-12</td>
      <td>20160619</td>
      <td>20160612</td>
    </tr>
  </tbody>
</table>
</div>



## Data Management and Representation
The day to day trends can be hard to analyze with so many values. During our exploratory data analysis, we will want to quickly locate large fluctiations in the price of coins. To target these, we will create a monthly and yearly change table.

First, create a pandas data frame for our master table. We will use this data frame to create our two new tables since pandas makes it easy to modify and add data. To begin, we will create two new columns, Month and Year. These columns will replace the Date column in the additional tables. We will also remove the time portion of the date column.

In some cases, seperating the month and year from a combined date could be considered tidying our data. Data is considered messy when there is more than one value stored in a single column, more than one observation in each row, or more than one set of observations in a table. With tidy data, every row should correspond to a distince observation for each column. Our data in master and in the tables we are creating now will be tidy since every row has a unique set of values for the behavior of a given coin on a given date. If, for example, high and low were stored in one column, say high/low with the value 'high/low', this data would be messy. To tidy this data we would need to split high/low into the two columns by the '/' in the string. Then we can perform computations on the two values without extra steps. Another example would be having the data for every coin in one row corresponding to a date. For more information on tidying data, check out this link: https://towardsdatascience.com/whats-tidy-data-how-to-organize-messy-datasets-in-python-with-melt-and-pivotable-functions-5d52daa996c9


```python
master = pd.read_sql("SELECT * FROM Master",conn)
master['Year'] = master['Date'].apply(lambda c: c[0:4])
master['Month'] = master['Date'].apply(lambda c: c[0:7])
master['Date'] = master['Date'].apply(lambda c: c[0:10])
```

Now that we have our data ready in pandas, we will use the groupby function to add each unique Year and Month with their respective Name and Symbol values. This will give us a base values that we can merge future columns with.


```python
tableVals = ['Year','Month']
newTables = [None,None]

for idx in range(0,2):
  newTables[idx] = master[[tableVals[idx],'Name','Symbol']].groupby([tableVals[idx],'Name','Symbol'], as_index=False).min()
  
newTables[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Name</th>
      <th>Symbol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013</td>
      <td>Bitcoin</td>
      <td>BTC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013</td>
      <td>Dogecoin</td>
      <td>DOGE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013</td>
      <td>Litecoin</td>
      <td>LTC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>XRP</td>
      <td>XRP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>Bitcoin</td>
      <td>BTC</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have our base values, lets merge the rest of the columns into our database. For high and low we can use groupby and merge to add the max and min values from each time period. Open and close is a little more complicated. We must use the max and min values for the Date for both month and year, since some coins were created in the middle of year or the month. Finally for Volume and Marketcap we can use the median values to represent the average among each year. Finally, we will also add a start and end column and two matching key columns to the data tables.


```python
for idx in range (0,2):
  #Add start and end date coluns
  newTables[idx]['startDate'] = newTables[idx].merge(master[[tableVals[idx],'Name','Date']].groupby([tableVals[idx],'Name'],as_index=False).min(),'inner')['Date']
  newTables[idx]['endDate'] = newTables[idx].merge(master[[tableVals[idx],'Name','Date']].groupby([tableVals[idx],'Name'],as_index=False).max(),'inner')['Date']
  #Add start and end date key columns
  newTables[idx]['startDateKey'] = newTables[idx]['startDate'].apply(lambda c: int(c.replace('-','')))
  newTables[idx]['endDateKey'] = newTables[idx]['endDate'].apply(lambda c: int(c.replace('-','')))
  #Add High Column with merge
  newTables[idx] = newTables[idx].merge(
      master[[tableVals[idx],'Name','High']].groupby([tableVals[idx],'Name'], as_index=False).max(),
      'inner')
  #Add Low Column with merge
  newTables[idx] = newTables[idx].merge(
      master[[tableVals[idx],'Name','Low']].groupby([tableVals[idx],'Name'], as_index=False).min(),
      'inner')
  #Add Open Column with merge
  openTbl = master[[tableVals[idx],'Name','Date']].groupby(
      [tableVals[idx],'Name'],as_index=False).min().merge(
          master[[tableVals[idx],'Name','Open','Date']],'left')
  newTables[idx] = newTables[idx].merge(openTbl[[tableVals[idx],'Name','Open']],'inner')
  #Add Close Column with merge
  closeTbl = master[[tableVals[idx],'Name','Date']].groupby(
      [tableVals[idx],'Name'],as_index=False).max().merge(
          master[[tableVals[idx],'Name','Close','Date']],'left')
  newTables[idx] = newTables[idx].merge(closeTbl[[tableVals[idx],'Name','Close']],'inner')
  #Add Average Volume  and Marketcap with merge
  for col in ['Volume','Marketcap']:
    newTables[idx] = newTables[idx].merge(master[[tableVals[idx],'Name',col]].groupby([tableVals[idx],'Name'],as_index=False).mean(),'inner')
    newTables[idx] = newTables[idx].rename(columns={col:f"Avg{col}"})

newTables[0].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Name</th>
      <th>Symbol</th>
      <th>startDate</th>
      <th>endDate</th>
      <th>startDateKey</th>
      <th>endDateKey</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>AvgVolume</th>
      <th>AvgMarketcap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2013</td>
      <td>Bitcoin</td>
      <td>BTC</td>
      <td>2013-04-29</td>
      <td>2013-12-31</td>
      <td>20130429</td>
      <td>20131231</td>
      <td>1156.140015</td>
      <td>65.526001</td>
      <td>134.444000</td>
      <td>754.010010</td>
      <td>5.667401e+05</td>
      <td>3.066581e+09</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2013</td>
      <td>Dogecoin</td>
      <td>DOGE</td>
      <td>2013-12-16</td>
      <td>2013-12-31</td>
      <td>20131216</td>
      <td>20131231</td>
      <td>0.001520</td>
      <td>0.000116</td>
      <td>0.000299</td>
      <td>0.000422</td>
      <td>1.081273e+05</td>
      <td>6.444175e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2013</td>
      <td>Litecoin</td>
      <td>LTC</td>
      <td>2013-04-29</td>
      <td>2013-12-31</td>
      <td>20130429</td>
      <td>20131231</td>
      <td>53.146999</td>
      <td>1.618620</td>
      <td>4.366760</td>
      <td>24.347000</td>
      <td>2.971116e+05</td>
      <td>1.504182e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2013</td>
      <td>XRP</td>
      <td>XRP</td>
      <td>2013-08-05</td>
      <td>2013-12-31</td>
      <td>20130805</td>
      <td>20131231</td>
      <td>0.061445</td>
      <td>0.002875</td>
      <td>0.005875</td>
      <td>0.027330</td>
      <td>3.254044e+03</td>
      <td>1.064678e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2014</td>
      <td>Bitcoin</td>
      <td>BTC</td>
      <td>2014-01-01</td>
      <td>2014-12-31</td>
      <td>20140101</td>
      <td>20141231</td>
      <td>1017.119995</td>
      <td>289.295990</td>
      <td>754.969971</td>
      <td>320.192993</td>
      <td>2.509365e+07</td>
      <td>6.782807e+09</td>
    </tr>
  </tbody>
</table>
</div>



Finally, lets add our tables to our sql database with the pandas to_sql function.


```python
for idx in range(0,2):
  try:
    newTables[idx].to_sql(f"{tableVals[idx]}lyChange", conn)
  except:
    print(f"Table {tableVals[idx]}lyChange Already Exists")
```

Now that we have all of our tables, there is one more column we want to add. All of the numerical data provided is great, but it is not enough for a financial analysis of the crypto-currency's performances. We will need to compute the percent change for each day. The formula is as follows:

> % Change = (Close - Open) / Close

This value shows how much the crypto-currency rose or fell over the given time period. This will be useful for later analysis, where we will want to see what the return on investment will be for certain coins over time.


```python
cur = conn.cursor()

tables = [x[0] for x in cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name!='Trends'").fetchall()]
for table in tables:
  #Add Change column if there is not one
  if (not ('Change',) in cur.execute(f"SELECT name FROM PRAGMA_TABLE_INFO('{table}')").fetchall())  :
    cur.execute(f"ALTER TABLE {table} ADD COLUMN Change REAL")
  #Set Change values if theyre Null
  if (cur.execute(f"SELECT SUM(Change) FROM {table}").fetchone()[0] == None):
    cur.execute(f"UPDATE {table} SET Change = (Close - Open) / Close")
cur.close()

pd.read_sql("SELECT * FROM Master",conn).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Symbol</th>
      <th>Date</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Marketcap</th>
      <th>DateKey</th>
      <th>Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-05 23:59:59</td>
      <td>55.112358</td>
      <td>49.787900</td>
      <td>52.675035</td>
      <td>53.219243</td>
      <td>0.000000e+00</td>
      <td>8.912813e+07</td>
      <td>20201005</td>
      <td>0.010226</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-06 23:59:59</td>
      <td>53.402270</td>
      <td>40.734578</td>
      <td>53.291969</td>
      <td>42.401599</td>
      <td>5.830915e+05</td>
      <td>7.101144e+07</td>
      <td>20201006</td>
      <td>-0.256839</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-07 23:59:59</td>
      <td>42.408314</td>
      <td>35.970690</td>
      <td>42.399947</td>
      <td>40.083976</td>
      <td>6.828342e+05</td>
      <td>6.713004e+07</td>
      <td>20201007</td>
      <td>-0.057778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-08 23:59:59</td>
      <td>44.902511</td>
      <td>36.696057</td>
      <td>39.885262</td>
      <td>43.764463</td>
      <td>1.658817e+06</td>
      <td>2.202651e+08</td>
      <td>20201008</td>
      <td>0.088638</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Aave</td>
      <td>AAVE</td>
      <td>2020-10-09 23:59:59</td>
      <td>47.569533</td>
      <td>43.291776</td>
      <td>43.764463</td>
      <td>46.817744</td>
      <td>8.155377e+05</td>
      <td>2.356322e+08</td>
      <td>20201009</td>
      <td>0.065216</td>
    </tr>
  </tbody>
</table>
</div>



## Exploratory Data Analysis
Now that we have our data stored correctly, lets see what information it might tell us. To bein, we will try to see what type of function might predict the behavior of a crypto currency the best. To tackle this, we wil compute linear regressions on the data for lines and polynomials. Plotting each of these regressions helps to determine whether or not it describes the data well. For each of the regressions, we will first compute it for the entire market, then for one specific crypto currency. 


```python
#Linear Regression On the whole market (Open and Close values with the start and end date keys)
cur = conn.cursor()
#Build lists of data
X = [[1., float(x[0])] for x in cur.execute("SELECT StartDateKey FROM MonthlyChange").fetchall()]
X = X + [[1., float(x[0])] for x in cur.execute("SELECT EndDateKey FROM MonthlyChange").fetchall()]
y = [float(x[0]) for x in cur.execute("SELECT Open FROM MonthlyChange").fetchall()]
y = y + [float(x[0]) for x in cur.execute("SELECT Close FROM MonthlyChange").fetchall()]
cur.close()

marketReg = LinearRegression(fit_intercept=True).fit(X,y)
print(f"Whole Market Regression: y = ({marketReg.coef_[0]})x0 + ({marketReg.coef_[1]})x1")

#Show predictions for the last 10 months
#create a plotable data table
mthChange = pd.read_sql("SELECT Name, Open as Val, StartDateKey as Date FROM MonthlyChange",conn)
mthChange = mthChange.append(pd.read_sql("SELECT Name, Close as Val, EndDateKey as Date FROM MonthlyChange",conn))
mthChange = mthChange.sort_values('Date', ignore_index=True)
mthChange['pred'] = mthChange['Date'].apply(lambda c: marketReg.predict([[1.,float(c)]])[0])
dates = mthChange['Date'].unique().tolist()
mthChange['idx'] = mthChange['Date'].apply(lambda c: dates.index(c))

ax = mthChange.plot(x='idx',y='Val',kind='scatter',c='b')
mthChange.plot(x='idx',y='pred',ax=ax,c='r')
```

    Whole Market Regression: y = (0.0)x0 + (0.025125605576467126)x1





    <matplotlib.axes._subplots.AxesSubplot at 0x7fdba3c52550>




![png](320_FINAL_files/320_FINAL_18_2.png)


It seems that the linear regression for the entire market does not describe all of the data well. This regression would tell us what the return on investment of evenly diversifying your portfolio into all coins would be. However, the regression does not seem to be a satisfactory description for the market.

Next, lets try using a linear regression on one coin. We will use ethereum, since it is popular and growing.


```python
coin = 'Ethereum'
eTbl = pd.read_sql(f"SELECT DateKey as Date, Open as Val FROM Master WHERE Name = '{coin}'",conn)
dates = eTbl['Date'].unique().tolist()
eTbl['idx'] = eTbl.index

#Compute Regression
X = [[1., float(x)] for x in eTbl['idx'].tolist()]
y = eTbl['Val'].tolist()
eReg = LinearRegression(fit_intercept=True).fit(X,y)

#Plot Data
eTbl['pred'] = eTbl['idx'].apply(lambda c: eReg.predict([[1.,float(c)]])[0])
ax = eTbl.plot(x='idx',y='Val',kind='scatter',c='b')
eTbl.plot(x='idx',y='pred',c='r',ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdba3054cd0>




![png](320_FINAL_files/320_FINAL_20_1.png)


This regression certainly seems to follow the trend of the data better than the entire market's, but it still does not capture the fluctuation in value. To attempt to see this, we will evaluate a multiple degree regression with degrees 1 to 5.


```python
#Multiple degree regression
mdTbl = pd.read_sql("SELECT Month as Date, Open as Val FROM MonthlyChange",conn)
dates = mdTbl['Date'].unique().tolist()
mdTbl['idx'] = mdTbl['Date'].apply(lambda c: dates.index(c))

X = [[1., float(x)] for x in mdTbl['idx'].tolist()]
y = mdTbl['Val'].tolist()

#Compute Regressions
for deg in range(1,6):
  model = Pipeline([('poly',pf(degree=deg)), ('lin',LinearRegression(fit_intercept=True))])
  reg = model.fit(X,y)
  mdTbl[f"{deg}-degree"] = mdTbl['idx'].apply(lambda c: reg.predict([[1.,float(c)]])[0])
  
#Create plot
ax = mdTbl[mdTbl.index % 2 == 0].plot(x='idx',y='Val',kind='scatter',c='b')
colors = ['r','g','orange','purple']
for idx in range(1,5):
  ax = mdTbl.plot(x='idx',y=f'{idx}-degree',c=colors[idx-1],ax=ax)

mdTbl.plot(x='idx',y='5-degree',c='y',ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdba4312f50>




![png](320_FINAL_files/320_FINAL_22_1.png)


After plotting the multiple degree regressions along with the first degree regression, it is clear that adding more complexity to the equation cannot handle the fluctuation in value for the entire market. In general, using a linear regression is not a good way to model the state of a market. The first degree regression is the least expensive and just as accurate as each higher degree regression. If there were a case where this equation were helpful, the single degree regression would be the best to use.

Now, lets see if a multiple degree regression works to describe the behavior of a single coin. Again, in this example, we will use Ethereum.


```python
coin = 'Ethereum'
#Gather data
eTbl = eTbl.rename(columns={'pred':'1-degree'})
dates = eTbl['Date'].unique().tolist()
X = [[1., float(x)] for x in eTbl['idx'].tolist()]
y = eTbl['Val'].tolist()

#Compute regressions
for deg in range(2,6):
  model = Pipeline([('poly',pf(degree=deg)), ('lin',LinearRegression(fit_intercept=True))])
  reg = model.fit(X,y)
  eTbl[f"{deg}-degree"] = eTbl['idx'].apply(lambda c: reg.predict([[1.,float(c)]])[0])

#Plot data
ax = eTbl.plot(x='idx',y='Val',kind='scatter',c='b')
for idx in range(1,5):
  ax = eTbl.plot(x='idx',y=f"{idx}-degree",c=colors[idx-1],ax=ax,)
eTbl.plot(x='idx',y='5-degree',c='y',ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fdbbaf73410>




![png](320_FINAL_files/320_FINAL_24_1.png)


In this plot, it seems that the 3rd - 5th degree regressions could do a good job at modeling the behavior of the single coin. However, we want to try and find a model that will be able to predict the future behavior of each coin. We will test this in a later section.


Many people speculate that crypto-currency fluctuates based on it's popularity at a given time. To explore these claims, lets try and plot the High data for one crypto currency versus the trends data, and see if there is a relationship. First we need to conver them both to months, and collect their data to plot.


```python
import matplotlib.pyplot as plt

monthly_high = pd.read_sql("SELECT Name, High, Month FROM MonthlyChange",conn)
months = []
for month in monthly_high['Month']:
  months.append(month.replace('-', ''))
monthly_high['MonthKey'] = months
model = LinearRegression().fit(monthly_high['MonthKey'][monthly_high['Name'] == 'Bitcoin'].values.reshape(-1, 1), 
                               monthly_high['High'][monthly_high['Name'] == 'Bitcoin'].values.reshape(-1, 1))

trends = []
i = 1
temp = 0.
for weeks in trends_df['Bitcoin']:
  if i == 4:
    trends.append(temp)
    i = 1
    temp = 0. 
  else:
    temp += weeks
    i = i + 1

months = monthly_high['MonthKey'][monthly_high['Name'] == 'Bitcoin']
trends_months = months[len(months) - len(trends):]
high_data = monthly_high['High'][monthly_high['Name'] == 'Bitcoin']

print(len(months))
print(len(high_data))

plt.scatter(months, high_data)
plt.scatter(trends_months, trends, color='red')
plt.legend(labels=['Market High', 'Google Searches'])
plt.title('Market High and Trend Searches vs Time (Un-normalized)')
plt.ylabel('Market Value & Number of Searches')
plt.xticks(ticks=[])
plt.xlabel('Years (2014 - 2021)')
plt.show()
```

    95
    95



![png](320_FINAL_files/320_FINAL_26_1.png)


As we can see, most values in the trends data is ranging 0-100, meaning when placed next to data in the thousands, we can not see any relationship, because the trends data is so small. The solution of this is to normalize both datasets by dividing each value in each dataset by their datasets maximum value. That is what we do below.


```python
# Normalize the data and look for trends again

max_high = max(high_data)
max_trend = max(trends)

norm_high_data = []
for v in high_data:
  norm_high_data.append(float(v) / max_high)

for ind, v in enumerate(trends):
  trends[ind] = float(v) / max_trend

plt.legend(labels=['Market High', 'Google Searches'])
plt.title('Market High and Trend Searches vs Time (Normalized)')
plt.ylabel('Market Value & Number of Searches')
plt.xticks(ticks=[])
plt.xlabel('Years (2014 - 2021)')
plt.scatter(months, norm_high_data)
plt.scatter(trends_months, trends, color='red')
plt.show()
```


![png](320_FINAL_files/320_FINAL_28_0.png)


As we can see now, there does seem to be some trend in market value and Google search trends. It appears that when the trends data spikes, a spike in market high value is imminent. This indicates that Google's trends data might just be a good indicator of stock price value, but one example is not enough to say for sure.

## Hypothesis Testing, Supervised Learning
With this data, we can create a Linear based sin regression based on the trend data, that predicts the high price. So we take the trends data as input, the market value as output and train a model to predict the high price given trends data. Training our model on Bitcoin, we can then test the models performance on another coin, for instance Dogecoin.


```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create our training data on Bitcoin
train_trends_input = []
train_high_label = []

# Convert trends data of Bitcoin to an array of Monthly totals
i = 1
temp = 0.
for weeks in trends_df['Bitcoin']:
  if i == 4:
    train_trends_input.append(temp)
    i = 1
    temp = 0. 
  else:
    temp += weeks
    i = i + 1

# Grab the high data from the table, and only relevant months
train_high_label = monthly_high['High'][monthly_high['Name'] == 'Bitcoin']
train_high_label = train_high_label[len(train_high_label) - len(train_trends_input):]

# Get the maximums of both lists for normalization
max_high = max(train_high_label)
max_trend = max(train_trends_input)

norm_X = []
norm_y = []

# Normalize both lists
for ind, v in enumerate(train_high_label):
  norm_y.append(v / max_high)

for ind, v in enumerate(train_trends_input):
  norm_X.append(v / max_trend)

# Create test data using Dogecoin, same process as above
train_trends_input = []
train_high_label = []

# Convert trends data of Dogecoin to an array of Monthly totals
i = 1
temp = 0.
for weeks in trends_df['Dogecoin']:
  if i == 4:
    train_trends_input.append(temp)
    i = 1
    temp = 0. 
  else:
    temp += weeks
    i = i + 1

# Grab the high data from the table, and only relevant months
train_high_label = monthly_high['High'][monthly_high['Name'] == 'Dogecoin']
train_high_label = train_high_label[len(train_high_label) - len(train_trends_input):]

# Get maximums for normalization
max_high = max(train_high_label)
max_trend = max(train_trends_input)

norm_X_test = []
norm_y_test = []

# Normalize to push data to [0, 1]
for ind, v in enumerate(train_high_label):
  norm_y_test.append(v / max_high)

for ind, v in enumerate(train_trends_input):
  norm_X_test.append(v / max_trend)

# training using Bitcoin data
regressor = LinearRegression()
regressor = regressor.fit([norm_X], [norm_y]) # fit the model for training data

# predict the target given the trends data
prediction_training_targets = regressor.predict([norm_X])

# Compute Accuracy scores
self_accuracy = regressor.score([norm_X], [norm_y])
print(f"Accuracy for training data: {self_accuracy}")

# Use model to predict on test data
prediction_test_targets = regressor.predict([norm_X_test])

# plot the predicted and actual target for test data
plt.plot(np.squeeze(prediction_test_targets), '-*')
plt.plot(norm_y_test, '-o' )
plt.legend(['prediction', 'actual'])
plt.title('Dogecoin Predicted vs Actual Market High Value')
plt.ylabel('Market value (Normalized)')
plt.xlabel('Months')
plt.show()

```

    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:434: FutureWarning: The default value of multioutput (not exposed in score method) will change from 'variance_weighted' to 'uniform_average' in 0.23 to keep consistent with 'metrics.r2_score'. To specify the default value manually and avoid the warning, please either call 'metrics.r2_score' directly or make a custom scorer with 'metrics.make_scorer' (the built-in scorer 'r2' uses multioutput='uniform_average').
      "multioutput='uniform_average').", FutureWarning)
    /usr/local/lib/python3.7/dist-packages/sklearn/metrics/_regression.py:582: UndefinedMetricWarning: R^2 score is not well-defined with less than two samples.
      warnings.warn(msg, UndefinedMetricWarning)


    Accuracy for training data: nan



![png](320_FINAL_files/320_FINAL_31_2.png)


Finally, we will test the ability of a multiple degree regression to predict the future behavior of a crypto currency. To accomplish this, we will take the beginning 70% of a coin's High data and fit a regression to this. Then we will test the regression on the last 30% of the High data and evaluate how well it fits.


```python
#Linear Regression Accuracy testing
coin = 'Cardano'
cTbl = pd.read_sql(f"SELECT DateKey as Date, High FROM Master WHERE Name='{coin}'",conn)
cTbl['idx'] = cTbl.index
cutOff = int(np.floor(.7 * len(cTbl)))
#Build test set
X = [[1., float(x)] for x in cTbl['idx'].tolist()][0:cutOff]
y = cTbl['High'].tolist()[0:cutOff]
regs = {}
for deg in range(1,6):
  model = Pipeline([('poly',pf(degree=deg)),('lin',LinearRegression())])
  regs[deg] = model.fit(X,y)

#Test the fit and plot
X_test = [[1., float(x)] for x in cTbl['idx'].tolist()][cutOff:]
y_test = cTbl['High'].tolist()[cutOff:]

colors.append('black')
trainData = cTbl[:cutOff]
testData = cTbl[cutOff:]
plt.plot(trainData['idx'].tolist(), trainData['High'].tolist(),c='g')
plt.plot(testData['idx'].tolist(), testData['High'].tolist(), c='r')
for deg in regs.keys():
  key = f"{deg}-degree"
  data = cTbl['idx'].apply(lambda c: regs[deg].predict([[1.,float(c)]])[0])
  cTbl[key] = data
  #Plot Prediction
  plt.plot(cTbl['idx'].tolist(), data, '--', c=colors[deg-1])

  #Score test data
  score = regs[deg].score(X_test, y_test)
  print(f"{key} regression accuracy (R^2 value): {score}")

plt.title("Cardano Regression Predictions")
plt.ylabel("High")
plt.xlabel("Index Value")
plt.legend(['Train','Test'] + list(regs.keys()))

#Plot the best Regression Model
plt.figure()
plt.plot(cTbl['idx'], cTbl['High'], c='b')
plt.plot(cTbl['idx'], cTbl['3-degree'], c='r')
plt.title("Cardano 3rd Degree Regression")
plt.legend(['High','Prediction'])
```

    1-degree regression accuracy (R^2 value): -1.4990953420121054
    2-degree regression accuracy (R^2 value): -0.21339431193948122
    3-degree regression accuracy (R^2 value): -4.631884966864126
    4-degree regression accuracy (R^2 value): -200.97313043643123
    5-degree regression accuracy (R^2 value): -815.9502469582515





    <matplotlib.legend.Legend at 0x7fdba4280490>




![png](320_FINAL_files/320_FINAL_33_2.png)



![png](320_FINAL_files/320_FINAL_33_3.png)


## Final Results/Findings
What we have found from our Trends Regression is a model that appears to predict a higher value than what is typically the market output. That being said, it does tend to increase prior to the market value actually increasing, which could be a useful model for making money. That being said, we could also take this model further, by training it on more data, and potentially decreasing the complexity, or hyperfocusing on the volatile sections, however for this case our model is good. It does seem though, that there is one occurence where our model predicts a spike that does not occur in the market data. So, there are flaws, but certainly hope.

On the other hand, for the basic linear regression model, relating value to time, there is no good representation of the data. In our example, the 3rd degree regression seems to describe the fluctuations in price generally, but the model has a R^2 score of less than zero. This means that the regression is less accurate than a horizontal line. So, if you were to base your investments off this model, you would be better off just keeping your money in the bank. Because of this, a regression relating time and value is not a good predictor for market behaviors.
