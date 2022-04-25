# -*- coding: utf-8 -*-
"""Stock Market Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gvHy5V-Re9VDZDydmFufzTNUaDh4yVtq

https://colab.research.google.com/
"""

import pandas
from pandas_datareader import DataReader

from datetime import datetime

stocks_list = ["FB",
               "AMZN",
               "NFLX",
               "GOOG"]

start = datetime(datetime.now().year - 1, 
                 datetime.now().month,
                 datetime.now().day)

end = datetime.now()

for stock in stocks_list:

  globals()[stock] = DataReader(stock,
                                "yahoo",
                                start,
                                end)

FB.describe()

print(AMZN)

NFLX.info()

GOOG["High"].plot()

GOOG["Volume"].plot(legend = True,
                    figsize = (14, 6))

moving_average_intervals = [5, 20, 50]

for moving_average in moving_average_intervals:

  column_name = "moving_average for %s days" %(str(moving_average))

  GOOG[column_name] = GOOG["Adj Close"].rolling(moving_average).mean()

GOOG[["Adj Close", 
      "moving_average for 5 days", 
      "moving_average for 20 days", 
      "moving_average for 50 days"]].plot(figsize=(14, 6))