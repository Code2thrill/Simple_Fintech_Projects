import pandas as pd
import requests as re
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

stock_columns = ['Ticker','Price','Price-to-Earnings Ratio','Shares to Buy']
dataframe = pd.DataFrame(columns=stock_columns)
for symbol_string in stock_symbols_strings:
    batch_api_call_url = f"https://sandbox.iexapis.com/stable/stock/market/batch/?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}"
    data = re.get(batch_api_call_url).json()
    for symbol in symbol_string.split(','):
        dataframe = dataframe.append(pd.Series([symbol,
                                                data[symbol]['quote']['latestPrice'],
                                                data[symbol]['quote']['peRatio'],
                                                'N/A'],
                                               index=stock_columns),
                                     ignore_index=True)
#pick stock with PE between 15 to 40
df = dataframe[(dataframe['Price-to-Earnings Ratio']>15)&(dataframe['Price-to-Earnings Ratio']<40)]
df.sort_values('Price-to-Earnings Ratio',inplace=True)
df = df.loc[:49]
df.reset_index(inplace=True)
df.drop('index', axis='columns', inplace=True)
# print(df)
def get_portfolio_value():
    global portfolio_value  #define a variable in a function as a global variable
    portfolio_value = input("Please enter your portfolio value: ")
    try:
        portfolio_value_float = float(portfolio_value)
    except ValueError:
        print("Not a number. Please try again.")
        portfolio_value = input("Please enter your portfolio value")
get_portfolio_value()
position_size = float(portfolio_value)/len(df.index)
for row in range(0, len(df['Ticker'])):
    shares_to_buy=math.floor(position_size/df['Price'][row])
    df.loc[row, 'Shares to Buy']=shares_to_buy
print(df)
