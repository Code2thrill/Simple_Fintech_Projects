import pandas as pd
import requests as re
import numpy as np
from scipy import stats
from statistics import mean
import math

tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
stocks_table = tables[0]
IEX_CLOUD_API_TOKEN = "Tpk_83d95ade22024ff1bb4460895342ca75"

def split_list(list, number_per_group):
    for index in range(0, len(list), number_per_group):
        yield list[index:index+number_per_group]

stock_symbols_groups = list(split_list(stocks_table['Symbol'], 100))

stock_symbols_strings = []
for index in range(0, len(stock_symbols_groups)):
    stock_symbols_strings.append(",".join(stock_symbols_groups[index]))

composite_columns = ['Ticker',
                     'Price',
                     'Shares to Buy',
                     'Price-to-Earnings Ratio',
                     'Price-to-Earnings Percentile',
                     'Price-to-Book Ratio',
                     'Price-to-Book Percentile',
                     'Price-to-Sales Ratio',
                     'Price-to-Sales Percentile',
                     'EV/EBITDA',
                     'EV/EBITDA Percentile',
                     'EV/Gross Profit',
                     'EV/Gross Profit Percentile',
                     'Robust Value Score']
composite_dataframe = pd.DataFrame(columns=composite_columns)
for symbol_string in stock_symbols_strings:
    batch_api_call_url = f"https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}"
    data = re.get(batch_api_call_url).json()
# print(data)
    for symbol in symbol_string.split(','):
        enterprise_value = data[symbol]['advanced-stats']['enterpriseValue']
        ebitda = data[symbol]['advanced-stats']['EBITDA']
        gross_profit=data[symbol]['advanced-stats']['grossProfit']
        try:
            ev_over_ebitda = enterprise_value/ebitda #由于可能不是所有的sotck都有这两个可计算的值所以用try回避错误
        except TypeError:
            ev_over_ebitda=np.NaN #numpy array type
        try:
            ev_over_gross_profit = enterprise_value/gross_profit
        except TypeError:
            ev_over_gross_profit=np.NaN #numpy array type
        composite_dataframe=composite_dataframe.append(pd.Series([
            symbol,
            data[symbol]['quote']['latestPrice'],
            'N/A',
            data[symbol]['quote']['peRatio'],
            'N/A',
            data[symbol]['advanced-stats']['priceToBook'],
            'N/A',
            data[symbol]['advanced-stats']['priceToSales'],
            'N/A',
            ev_over_ebitda,
            'N/A',
            ev_over_gross_profit,
            'N/A',
            'N/A'
        ],index=composite_columns),
        ignore_index=True)

#data processing
# print(composite_dataframe[composite_dataframe.isnull().any(axis='columns')])
#notice there are lots of none values in the dataframe.
#one of the solutions is to fill NA values with column value average using df[column].fillna()
for column in ['Price-to-Earnings Ratio',
               'Price-to-Book Ratio',
               'Price-to-Sales Ratio',
               'EV/EBITDA',
               'EV/Gross Profit']:
    composite_dataframe[column].fillna(composite_dataframe[column].mean(),inplace=True)
# print(composite_dataframe[composite_dataframe.isnull().any(axis='columns')])
# print(composite_dataframe)#notice the "N/A" as a string won't be affected by the .fillna()

#fill in the columns with percentile values
ratios_and_percentiles={'Price-to-Earnings Ratio': 'Price-to-Earnings Percentile',
                        'Price-to-Book Ratio': 'Price-to-Book Percentile',
                        'Price-to-Sales Ratio': 'Price-to-Sales Percentile',
                        'EV/EBITDA': 'EV/EBITDA Percentile',
                        'EV/Gross Profit': 'EV/Gross Profit Percentile'}
for row in composite_dataframe.index:
    for ratio in ratios_and_percentiles.keys():
        composite_dataframe.loc[row,ratios_and_percentiles[ratio]]=stats.percentileofscore(
            composite_dataframe[ratio],
            composite_dataframe.loc[row,ratio]
        )

for row in composite_dataframe.index:
    percentiles=[]
    for ratio in ratios_and_percentiles.keys():
        percentiles.append(composite_dataframe.loc[row,
                           ratios_and_percentiles[ratio]])
    composite_dataframe.loc[row,'Robust Value Score']=mean(percentiles)

composite_dataframe.sort_values(by="Robust Value Score", ascending=False, inplace=True)
composite_dataframe = composite_dataframe[:50]
composite_dataframe.reset_index(drop=True, inplace=True)

def get_portfolio_value():
    global portfolio_value  #define a variable in a function as a global variable
    portfolio_value = input("Please enter your portfolio value: ")
    try:
        portfolio_value_float = float(portfolio_value)
    except ValueError:
        print("Not a number. Please try again.")
        portfolio_value = input("Please enter your portfolio value")
get_portfolio_value()
position_size = float(portfolio_value)/len(composite_dataframe.index)
for row in range(0, len(composite_dataframe['Ticker'])):
    composite_dataframe.loc[row,'Shares to Buy']=math.floor(position_size / composite_dataframe['Price'][row])
print(composite_dataframe)
import xlsxwriter
excel_writer=pd.ExcelWriter('top50 stocks with composites valuation.xlsx', engine='xlsxwriter')
composite_dataframe.to_excel(excel_writer,
                             sheet_name="composites Strategy",
                             index=False)
excel_writer.save()