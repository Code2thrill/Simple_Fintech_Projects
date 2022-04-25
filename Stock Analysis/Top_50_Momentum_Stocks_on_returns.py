import momentum as momentum
import pandas as pd
import requests as re
from scipy import stats
from statistics import mean

tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
stocks_table = tables[0]

IEX_CLOUD_API_TOKEN = "Tpk_83d95ade22024ff1bb4460895342ca75"

#split S&P 500 symbol list into batches and put them into a list
def split_list(list, number_per_group):
    for index in range(0, len(list), number_per_group):
        yield list[index:index+number_per_group]
stock_symbols_groups = list(split_list(stocks_table['Symbol'], 100))

#把stock_symbols_groups的list里的6组symbol strings，用"，"拼接在一起的6组strings
stock_symbols_strings = []
for index in range(0, len(stock_symbols_groups)):
    stock_symbols_strings.append(",".join(stock_symbols_groups[index]))

#create a empty dataframe
stock_columns = ["Ticker", "Price", "One-Year Price Return", "Share to Buy"]
stocks_dataframe = pd.DataFrame(columns=stock_columns)

#request batch API calls for stocks' info with batched symbol list
for symbol_string in stock_symbols_strings:
    batch_api_call_url = f"https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}"
    data = re.get(batch_api_call_url).json()
    for symbol in symbol_string.split(","):
        stocks_dataframe = stocks_dataframe.append(pd.Series([symbol,
                                                              data[symbol]["quote"]["latestPrice"],
                                                              data[symbol]["stats"]["year1ChangePercent"],
                                                              "N/A"],
                                                             index=stock_columns),
                                                   ignore_index=True)
#Remove Low Momentum Stocks:

#sort returns first; put stocks with most returns on the top
stocks_dataframe.sort_values("One-Year Price Return", ascending=False, inplace=True)
#pick top 50 stocks with best returns within a year
stocks_dataframe = stocks_dataframe[:50]
#reset dataframe index so index will reset from 0 to 49
stocks_dataframe.reset_index(drop=True, inplace=True)
#stock with greater one-year return are the ones have greater momentum
def get_portfolio_value():
    global portfolio_value  #define a variable in a function as a global variable
    portfolio_value = input("Please enter your portfolio value: ")
    try:
        portfolio_value_float = float(portfolio_value)
    except ValueError:
        print("Not a number. Please try again.")
        portfolio_value = input("Please enter your portfolio value")
get_portfolio_value()
#assume portfolio equally distributed to the top 50 stocks;把钱平均分成50份
position_size = float(portfolio_value)/len(stocks_dataframe.index)
#每份钱除以股票价格，得股票数量shares_to_buy
import math
for index in range(0, len(stocks_dataframe['Ticker'])):
    share_to_buy = math.floor(position_size/stocks_dataframe['Price'][index])
    #floor is round float to integer
    stocks_dataframe.loc[index, 'Share to Buy'] = share_to_buy
print(stocks_dataframe)

# Find High Quality Momentum Stocks
high_quality_momentum_columns = [
    'Ticker',
    'Price',
    'shares_to_buy',
    'One-Year Price Return',
    'One-Year Return Percentile',
    'Six-Month Price Return',
    'Six-Month Return Percentile',
    'Three-Month Price Return',
    'Three-Month Return Percentile',
    'One-Month Price Return',
    'One-Month Return Percentile',
    'High Quality Momentum Score'
]
high_quality_momentum_dataframe = pd.DataFrame(columns=high_quality_momentum_columns)

for symbol_string in stock_symbols_strings:
    batch_api_call_url = f"https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}"
    data = re.get(batch_api_call_url).json()
    for symbol in symbol_string.split(","):
        high_quality_momentum_dataframe = high_quality_momentum_dataframe.append(pd.Series([
            symbol,
            data[symbol]['quote']['latestPrice'],
            'N/A',
            data[symbol]['stats']['year1ChangePercent'],
            'N/A',
            data[symbol]['stats']['month6ChangePercent'],
            'N/A',
            data[symbol]['stats']['month3ChangePercent'],
            'N/A',
            data[symbol]['stats']['month1ChangePercent'],
            'N/A',
            'N/A'
        ],index=high_quality_momentum_columns),#use index as high_quality_momentum_columns for pd.Series()
            ignore_index=True)#ignore the index for the high_quality_momentum_dataframe
        #以上的这两个index设定很巧妙的把pandas.Series的index变成了panda dataframe的column names

time_intervals = ["One-Year", "Six-Month", "Three-Month", "One-Month"]
for row in high_quality_momentum_dataframe.index:
    for time_interval in time_intervals:
        #passing column and row argument to stats.percentileofscore(), but they must fill in directly unless it will create new rows
        # column = high_quality_momentum_dataframe[f'{time_interval} Price Return']
        # row = high_quality_momentum_dataframe.loc[row, f'{time_interval} Price Return']
        high_quality_momentum_dataframe.loc[row,f"{time_interval} Return Percentile"] = stats.percentileofscore(
            high_quality_momentum_dataframe[f'{time_interval} Price Return'],
            high_quality_momentum_dataframe.loc[row, f'{time_interval} Price Return']
        )/100 #divided by 100 to get decimal
#find the 50 best momentum stocks
for row in high_quality_momentum_dataframe.index:
    momentum_percentiles = []
    for time_interval in time_intervals:
        momentum_percentiles.append(high_quality_momentum_dataframe.loc[
            row, f'{time_interval} Return Percentile'
                                   ])
    # print(momentum_percentiles) 生成了500个 ["One-Year return percentiles", "Six-Month return percentiles", "Three-Month return percentiles", "One-Month return percentiles"]
    high_quality_momentum_dataframe.loc[row,'High Quality Momentum Score']=mean(momentum_percentiles)#生成500个mean
#sort and pick top 50 high quality momentum stocks
high_quality_momentum_dataframe.sort_values(by='High Quality Momentum Score',
                                            ascending=False,#in descending order
                                            inplace=True)
top_50_high_quality = high_quality_momentum_dataframe[:50]
#reset index numbers
top_50_high_quality.reset_index(drop=True, inplace=True)
position_size_2 = float(portfolio_value)/len(high_quality_momentum_dataframe.index)#len of the df index list
#如果没有上一步的reset_index,下面的这个for loop会出现问题
for index in range(0, len(top_50_high_quality['Ticker'])):
    shares_to_buy=math.floor(position_size/top_50_high_quality['Price'][index])
    top_50_high_quality.loc[index, 'shares_to_buy']=shares_to_buy
# print(top_50_high_quality)
import xlsxwriter
excel_writer=pd.ExcelWriter('quantitative_momentum_stocks.xlsx', engine='xlsxwriter')
top_50_high_quality.to_excel(excel_writer,
                             sheet_name="Quantitative momentum Strategy",
                             index=False)
excel_writer.save()