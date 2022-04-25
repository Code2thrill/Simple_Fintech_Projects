import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from pandas_datareader import DataReader
from datetime import datetime
import plotly.graph_objs as go
import pandas as pd
import numpy
import plotly.express as px

app = dash.Dash()
page = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
wiki_df = page[0]
options = []
for element in wiki_df.index:
    options.append({'label':wiki_df['Security'][element], #get every row of the 'Name' column as key:label
                    'value':wiki_df['Symbol'][element]})

app.layout = html.Div([
    html.H1('Stock Dashboard'),
    html.Div([html.H3('Enter a stock symbol:',
                      style={'paddingRight':'30px'}),
              dcc.Input(id='my_stock_picker',
                        value="GOOG",
                        style={'fontSize':24, 'width':75}
                        )],style={'display':'inline-block',
                                  'verticalAlign':'top'}),
    html.Div([html.H3('Select a start and end date:'),
              dcc.DatePickerRange(id='my_date_picker',
                                  min_date_allowed=datetime(2004,1,1),
                                  max_date_allowed=datetime.today(),
                                  start_date = datetime(2010,1,1),
                                  end_date = datetime.today())
              ],style={'display':'inline-block'}),
    html.Div([html.Button(id='submit-button',
                          n_clicks=0,
                          children='submit',
                          style={'fontSize':26,'marginLeft':'30px'})
              ],style={'display':'inline-block'}),
    dcc.Graph(id='stock_graph'),
    html.Div([html.H2('Select multiple Stocks'),
              dcc.Dropdown(id='dropdown',
                           options = options,
                           value= ['GOOG','AAPL'],
                           multi=True)],style={'display':'inline-block'}),
    dcc.Graph(id='heatmap'),
    html.H2('Monte Carlo Analysis'),
    html.Div([html.H3('Select a stock'),
             dcc.Dropdown(id='dropdown2',
                          options = options,
                          value = ['GOOG'])],
             style={'display':'inline-block'}),
    html.Div([html.H3('Enter a starting price'),
        dcc.Input(id='start_price',
                        value=0,
                        style={'fontSize':24, 'width':75}
                        )],style={'display':'inline-block',
                                  'verticalAlign':'top'}),
    html.Div([html.Button(id='submit-button2',
                          n_clicks=0,
                          children='submit',
                          style={'fontSize':26,'marginLeft':'30px'})
              ],style={'display':'inline-block'}),
    dcc.Graph(id='monte_carlo_analysis')
])


@app.callback(Output('stock_graph','figure'),
              [Input('submit-button','n_clicks')],
              [State('my_stock_picker','value'),
               State('my_date_picker','start_date'),
               State('my_date_picker','end_date')])

def update_graph(n_clicks,my_stock_picker,start_date,end_date):
    start = datetime.strptime(start_date[:10],"%Y-%m-%d")
    end = datetime.strptime(end_date[:10],"%Y-%m-%d")
    df = DataReader(my_stock_picker,'yahoo', start ,end)
    moving_average_intervals = [20,100]
    for moving_average in moving_average_intervals:
        column_name = 'moving_average for %s days'%(str(moving_average))#fill % with str(moving_average)
        df[column_name] = df['Adj Close'].rolling(moving_average).mean()
    trace1 = go.Scatter(x = df.index, y= df['Adj Close'], mode='lines', name='Daily Close Price')
    trace2 = go.Scatter(x = df.index, y= df['moving_average for 20 days'], mode='lines', name='moving_average for 20 days')
    trace3 = go.Scatter(x = df.index, y= df['moving_average for 100 days'],mode='lines', name='moving_average for 100 days')
    data = [trace1, trace2, trace3]
    fig = {'data':data,
           'layout':{'title':'my_stock_picker'}}
    return fig

@app.callback(Output('heatmap','figure'),
              [Input('dropdown','value')])
def update_heatmap(stocks):#past five year returns cor
    start = datetime(datetime.now().year-5,datetime.now().month,datetime.now().day)
    end = datetime.now()
    close_df = DataReader(stocks,'yahoo', start ,end)['Adj Close']
    stock_returns = close_df.pct_change()
    z = stock_returns.corr()
    fig ={'data':[go.Heatmap(
        z=z,
        x=stocks,
        y=stocks,
        colorscale='haline')],
        'layout':go.Layout(
            title='5-year daily return correlations')}
    return fig


@app.callback(Output('monte_carlo_analysis','figure'),
              [Input('submit-button2','n_clicks')],
              [State('dropdown2','value'),
               State('start_price','value')])
def update_monte_graph(number_of_clicks,stock,starting_price):
    # fig = {'data':[{'x':[1,2],'y':[3,1]}],
    #        'layout':{'title':"TEST"}}
    start_price = starting_price
    days = 365
    dt = 1/days
    start = datetime(datetime.now().year-1,datetime.now().month,datetime.now().day)
    end = datetime.now()
    close_df = DataReader(stock,'yahoo', start ,end)['Adj Close']
    stock_returns = close_df.pct_change()
    sigma = stock_returns.std()
    mu = stock_returns.mean()
    def monte_carlo_analysis(start_price,days,mu,sigma):
        price = numpy.zeros(days)
        price[0]=start_price
        shock = numpy.zeros(days)
        drift = numpy.zeros(days)
        for day in range(1,days):
            shock[day]= numpy.random.normal(
                loc = mu * dt,
                scale = sigma * numpy.sqrt(dt))
            drift[day]= mu * dt
            price[day]=price[day-1]+(price[day-1]*(drift[day]+shock[day]))
        return price
    annual_prices =[]
    for run in range(100):
        price = monte_carlo_analysis(start_price, days, mu, sigma)
        annual_prices.append(price)
    numpy_array = numpy.array(annual_prices)
    df_test = pd.DataFrame(numpy_array)
    data = [go.Scatter(x=df_test.columns,y=df_test.loc[a],mode='lines')for a in df_test.index]
    fig = {'data':data,
           'layout':{'title':'Expected Price'}}
    return fig


if __name__ == '__main__':
    app.run_server()