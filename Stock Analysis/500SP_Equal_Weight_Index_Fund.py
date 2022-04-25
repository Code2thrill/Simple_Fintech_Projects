import pandas
import requests as re
import pandas as pd

IEX_CLOUD_API_TOKEN = "Tpk_83d95ade22024ff1bb4460895342ca75"
page = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
stocks_table = page[0]
# stocks_columns = ['Ticker', 'Price', 'Market Capitalization', 'Shares to Buy']
# stocks_dataframe = pd.DataFrame(columns=stocks_columns)
# for symbol in stocks_table['Symbol']:
#     api_url = f"https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}"
#     data = re.get(api_url).json()
#     stocks_dataframe = stocks_dataframe.append(pd.Series([symbol,
#                                                 data['latestPrice'],
#                                                 data['marketCap'],
#                                                 "N/A"],
#                                                index=stocks_columns),
#                                                ignore_index=True)
# print(stocks_dataframe)
#make batch API calls
#下面的这个方法太棒了，可以有效的把一个长list分割成小段;另外yield用的很巧妙
def split_list(list, number_per_group):
    for index in range(0,len(list),number_per_group):
        yield list[index:index + number_per_group]
stock_symbols_groups = list(split_list(stocks_table['Symbol'],100))

stock_symbols_strings = []
for index in range(0,len(stock_symbols_groups)):
    stock_symbols_strings.append(",".join(stock_symbols_groups[index]))

stocks_columns = ['Ticker', 'Price', 'Market Capitalization', 'Shares to Buy']
batch_stocks_dataframe= pd.DataFrame(columns=stocks_columns)
for symbol_string in stock_symbols_strings:
    batch_api_call_url = f"https://sandbox.iexapis.com/stable/stock/market/batch?&token={IEX_CLOUD_API_TOKEN}&types=quote&symbols={symbol_string}"
    data = re.get(batch_api_call_url).json()

    for symbol in symbol_string.split(","):
        batch_stocks_dataframe = batch_stocks_dataframe.append(
            pd.Series([symbol,
                       data[symbol]['quote']['latestPrice'],
                       data[symbol]['quote']['marketCap'],
                       'N/A'],
                      index=stocks_columns), ignore_index=True)

# print(batch_stocks_dataframe)
portfolio_value = input("Enter your portfolio value: ")
try:
    portfolio_value_float = float(portfolio_value)
except ValueError:
    print("Not a number")
    portfolio_value = input("Enter your portfolio value: ")
import math
position_size = float(portfolio_value)/len(batch_stocks_dataframe.index)
for index in range(0,len(batch_stocks_dataframe["Ticker"])-1):
    batch_stocks_dataframe.loc[index, "Shares to Buy"] = math.floor(position_size/batch_stocks_dataframe["Price"][index])
#The Math.floor() function returns the largest integer less than or equal to a given number.
print(batch_stocks_dataframe)
import xlsxwriter
excel_writer = pd.ExcelWriter("stocks.xlsx", engine="xlsxwriter")
batch_stocks_dataframe.to_excel(excel_writer,
                                sheet_name="S&P 500 Stocks",
                                index=False)
excel_writer.save()

