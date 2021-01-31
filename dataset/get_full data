import pandas as pd
spy_df = pd.read_csv('dataset/SPY.csv')
smlv_df = pd.read_csv('dataset/SMLV.csv')
lng_df = pd.read_csv('dataset/LNG.csv')
def ddd(i):
    i['Date'] = pd.to_datetime(i['Date'])
    i = i.set_index('Date')
    i = i['Adj_Close'].to_frame()
    return i
lng_df = ddd(lng_df)
lng_df.rename(columns={'Adj_Close':'LNG'}, inplace=True)
smlv_df = ddd(smlv_df)
smlv_df.rename(columns={'Adj_Close':'SMLV'}, inplace=True)
spy_df = ddd(spy_df)
spy_df.rename(columns={'Adj_Close':'SPY'}, inplace=True)
# Join 3 stock dataframes together
full_df = pd.concat([lng_df, spy_df, smlv_df], axis=1).dropna()
full_df.to_csv('fulldata.csv')
