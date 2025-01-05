from tvDatafeed import TvDatafeed, Interval
#username = 'abcd@123'
#password = '1234'
#tv= TvDatafeed(username, password)

tv = TvDatafeed()
#index
nifty_index_data = tv.get_hist(symbol = 'NIFTY', exchange='NSE', interval = Interval.in_1_minute, n_bars=10000)

#Futures
#nifty_futures_data = tv.get_hist(symbol = 'NIFTY', exchange='NSE', interval = Interval.in_1_minute, n_bars=1000, fut_contract=1)

#Crudeoil
#crudeoil_data = tv.get_hist(symbol = 'CRUDEOIL', exchange='MCX', interval = Interval.in_1_minute, n_bars=1000, fut_contract=1)

#downloading data for extended market hours
#extended_price_data = tv.get_hist(symbol = 'EICHERMOT', exchange='NSE', interval = Interval.in_1_hour, n_bars=1000, extended_session=False)
print(nifty_index_data)

'''
Interval.in_1_minute

Interval.in_3_minute

Interval.in_5_minute

Interval.in_15_minute

Interval.in_30_minute

Interval.in_45_minute

Interval.in_1_hour

Interval.in_2_hour

Interval.in_3_hour

Interval.in_4_hour

Interval.in_daily

Interval.in_weekly

Interval.in_monthly'''





