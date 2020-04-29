#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
说明：
1.数据来源于天池比赛https://tianchi.aliyun.com/competition/entrance/231573/information
2.使用时间序列自回归
"""


# In[5]:


import matplotlib.pyplot as plt
import pandas as pd
#取出2014-02-01到2014-07-31每日申购总金额
def generate_purchase_seq():
    dateparse = lambda dates: pd.datetime.strptime(dates,"%Y%m%d")
    user_balance = pd.read_csv(r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\user_balance_table.csv",parse_dates=["report_date"],index_col="report_date",date_parser=dateparse)
    
    df = user_balance.groupby(["report_date"])["total_purchase_amt"].sum()
    
    purchase_seq = pd.Series(df,name="value")
    purchase_seq_201402_201407 = purchase_seq["2014-02-01":"2014-07-31"]
    purchase_seq_201402_201407.to_csv(path=r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_201402_201407.csv",header=True)    
generate_purchase_seq()

#将生成的purchase_seq_201402_201407,绘图
def purchase_seq_display(timeseries):
    graph = plt.figure(figsize=(10,4))
    ax = graph.add_subplot(111)
    ax.set(title="Total_Purchase_Amt",ylabel="Unit(yuan)",xlabel="Date")
    plt.plot(timeseries)
    plt.show()
    
dateparse = lambda dates: pd.datetime.strptime(dates,"%Y-%m-%d")
purchase_seq_201402_201407 = pd.read_csv(r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_201402_201407.csv", parse_dates=['report_date'],index_col='report_date', date_parser=dateparse)
purchase_seq_display(purchase_seq_201402_201407)   
 
    
#时间序列分解
from statsmodels.tsa.seasonal import seasonal_decompose

def decomposing(timeseries):
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.figure(figsize=(16,12))
    plt.subplot(411)
    plt.plot(timeseries,label="Original")
    plt.legend(loc="best")
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonarity')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residual')
    plt.legend(loc='best')
    plt.show()
    
dateparse = lambda dates: pd.datetime.strptime(dates,"%Y-%m-%d")
purchase_seq_201402_201407 = pd.read_csv(r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_201402_201407.csv", parse_dates=['report_date'],index_col='report_date', date_parser=dateparse)
decomposing(purchase_seq_201402_201407)

#采用ADF检验时间序列的平稳性，基于随机游走思想
from statsmodels.tsa.stattools import adfuller as ADF

def diff(timeseries):
    timeseries_diff1 = timeseries.diff(1)
    timeseries_diff2 = timeseries_diff1.diff(1)
    
    timeseries_diff1 = timeseries_diff1.fillna(0)
    timeseries_diff2 = timeseries_diff2.fillna(0)
    
    timeseries_adf = ADF(timeseries['value'].tolist())
    timeseries_diff1_adf = ADF(timeseries_diff1['value'].tolist())
    timeseries_diff2_adf = ADF(timeseries_diff2['value'].tolist())
    
    print('timeseries_adf : ', timeseries_adf)
    print('timeseries_diff1_adf : ', timeseries_diff1_adf)
    print('timeseries_diff2_adf : ', timeseries_diff2_adf)
    
    plt.figure(figsize=(16, 12))
    plt.plot(timeseries, label='Original', color='blue')
    plt.plot(timeseries_diff1, label='Diff1', color='red')
    plt.plot(timeseries_diff2, label='Diff2', color='purple')
    plt.legend(loc='best')
    plt.show()
    
diff(purchase_seq_201402_201407)
#对于原数据p值不显著，接受原假设，表明系数（单位根）不为0，存在，因此非平稳
#对于一阶差分，p值非常小，且t统计量小于p值为1%的统计检验，因此可以推断平稳

#基于ACF、PACF的自相关检验
import statsmodels.api as sm
#画出随机噪声（残差）相关图
def autocorrelation(timeseries, lags):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    sm.graphics.tsa.plot_acf(timeseries, lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    sm.graphics.tsa.plot_pacf(timeseries, lags=lags, ax=ax2)
    plt.show()
    
purchase_seq_201402_201407_diff1 = purchase_seq_201402_201407.diff(1)
purchase_seq_201402_201407_diff1 = purchase_seq_201402_201407_diff1.fillna(0)
autocorrelation(purchase_seq_201402_201407_diff1, 20)
#蓝色阴影部分为95%的置信区间，原假设为间隔为k的自相关系数为0

#使用ARIMA模型，首先对时间序列进行分解（STL），再采用ARIMA模型进行拟合趋势序列与残差序列
user_balance = pd.read_csv(r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\user_balance_table.csv")

df_tmp = user_balance.groupby(['report_date'])['total_purchase_amt', 'total_redeem_amt'].sum()
df_tmp.reset_index(inplace=True)

df_tmp['report_date'] = pd.to_datetime(df_tmp['report_date'], format='%Y%m%d')
df_tmp.index = df_tmp['report_date']

total_purchase_amt = plt.figure(figsize=(10, 4))
ax = total_purchase_amt.add_subplot(111)
ax.set(title='Total_Purchase_Amt',
       ylabel='Unit (yuan)', xlabel='Date')
plt.plot(df_tmp['report_date'], df_tmp['total_purchase_amt'])
plt.show()

#根据序列平稳性，选取2014-04-01~2014-07-31作为训练集，将2014-08-01~2014-08-10作为测试集
def generate_purchase_seq():
    dateparse = lambda dates: pd.datetime.strptime(dates, "%Y%m%d")
    user_balance = pd.read_csv(r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\user_balance_table.csv", parse_dates=['report_date'],
                               index_col="report_date", date_parser=dateparse)

    df = user_balance.groupby(["report_date"])["total_purchase_amt"].sum()
    purchase_seq = pd.Series(df, name="value")

    purchase_seq_train = purchase_seq['2014-04-01':'2014-07-31']
    purchase_seq_test = purchase_seq['2014-08-01':'2014-08-10']

    purchase_seq_train.to_csv(path=r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_train.csv", header=True)
    purchase_seq_test.to_csv(path=r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_test.csv", header=True)


generate_purchase_seq()

#查看训练集差分情况，并进行ADF检验
purchase_seq_train = pd.read_csv(r"C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_train.csv", parse_dates=['report_date'],
                                 index_col='report_date', date_parser=dateparse)
diff(purchase_seq_train)

decomposition = seasonal_decompose(purchase_seq_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

diff(trend)
diff(residual)
#从结果来看，趋势与残差都已经平稳，不需要差分

autocorrelation(residual, 20)
#根据结果，ACF有3阶截尾，PACF有3阶结尾

#通过网格搜索，使用AIC准则对p、q进行选择
import statsmodels.api as sm
#趋势函数
trend_evaluate = sm.tsa.arma_order_select_ic(trend, ic=["aic", "bic"], trend="nc", max_ar=4,
                                            max_ma=4)
print("trend AIC", trend_evaluate.aic_min_order)
print("trend BIC", trend_evaluate.bic_min_order)
#残差函数
residual_evaluate = sm.tsa.arma_order_select_ic(residual, ic=['aic', 'bic'], trend='nc', max_ar=4,
                                            max_ma=4)
print('residual AIC', residual_evaluate.aic_min_order)
print('residual BIC', residual_evaluate.bic_min_order)


from statsmodels.tsa.arima_model import ARIMA
def ARIMA_Model(timeseries, order):
    model = ARIMA(timeseries, order=order)
    return model.fit(disp=0)


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
purchase_seq_train = pd.read_csv(r'C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_train.csv', parse_dates=['report_date'],
                                 index_col='report_date', date_parser=dateparse)

purchase_seq_test = pd.read_csv(r'C:\Users\ASUS\Desktop\data\Purchase Redemption Data\purchase_seq_test.csv', parse_dates=['report_date'],
                                index_col='report_date', date_parser=dateparse)


decomposition = seasonal_decompose(purchase_seq_train)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

trend = trend.fillna(0)
seasonal = seasonal.fillna(0)
residual = residual.fillna(0)

# 趋势序列模型训练
trend_model = ARIMA_Model(trend, (1, 0, 0))
trend_fit_seq = trend_model.fittedvalues
trend_predict_seq = trend_model.predict(start='2014-08-01', end='2014-08-10', dynamic=True)

# 残差序列模型训练
residual_model = ARIMA_Model(residual, (2, 0, 1))
residual_fit_seq = residual_model.fittedvalues
residual_predict_seq = residual_model.predict(start='2014-08-01', end='2014-08-10', dynamic=True)

# 拟合训练集
fit_seq = pd.Series(seasonal['value'], index=seasonal.index)
fit_seq = fit_seq.add(trend_fit_seq, fill_value=0)
fit_seq = fit_seq.add(residual_fit_seq, fill_value=0)

plt.plot(fit_seq, color='red', label='fit_seq')
plt.plot(purchase_seq_train, color='blue', label='purchase_seq_train')
plt.legend(loc='best')
plt.show()

# 预测测试集
# 这里测试数据的周期性是根据seasonal对象打印的结果，看到里面的数据每7天一个周期，2014-08-01~2014-08-10的数据正好和2014-04-04~2014-04-13的数据一致
seasonal_predict_seq = seasonal['2014-04-04':'2014-04-13']

predict_dates = pd.Series(
    ['2014-08-01', '2014-08-02', '2014-08-03', '2014-08-04', '2014-08-05', '2014-08-06', '2014-08-07', '2014-08-08',
     '2014-08-09', '2014-08-10']).apply(lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d'))

seasonal_predict_seq.index = predict_dates

predict_seq = pd.Series(seasonal_predict_seq['value'], index=seasonal_predict_seq.index)
predict_seq = predict_seq.add(trend_predict_seq, fill_value=0)
predict_seq = predict_seq.add(residual_predict_seq, fill_value=0)

plt.plot(predict_seq, color='red', label='predict_seq')
plt.plot(purchase_seq_test, color='blue', label='purchase_seq_test')
plt.legend(loc='best')
plt.show()

#从结果来看，模型拟合训练集的效果还是不错的；在测试集上，模型基本上预测了序列的趋势和波动。实际上，这样的数据集不适合用 ARIMA 模型来拟合（序列的（线性）自相关性不强，受随机噪声影响较大），
#但是采用了时间序列分解的方法，暂且预测了一个序列的趋势。

