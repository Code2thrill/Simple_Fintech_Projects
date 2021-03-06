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