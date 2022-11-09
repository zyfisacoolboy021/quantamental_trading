#Code for Constructing Portfolio of 2015:
 
import pandas as pd
import numpy as np
import os
#coding = utf-8

path = "C://Users//18056//cis//"
df2 = pd.read_excel(path+"second.xlsx")
df2["code"] = df2["code"].astype(str).str.pad(6,fillchar="0")
df2.head()

df = pd.read_excel(path+"first.xlsx")
q42014 = pd.read_excel(path+"2014q4.xlsx")
df = df.merge(q42014,on="code")x
df["code"] = df["code"].astype(str).str.pad(6,fillchar="0")

df2012_2014 = pd.concat([df,df2])
df_withoutna = df2012_2014.dropna(how="any")
df_withoutextreme = df_withoutna.drop(
    df_withoutna[(abs(df_withoutna["2012Q4"])>0.85)|
                 (abs(df_withoutna["2013Q1"])>0.85)|
                 (abs(df_withoutna["2013Q2"])>0.85)|
                 (abs(df_withoutna["2013Q3"])>0.85)|
                 (abs(df_withoutna["2013Q4"])>0.85)|
                 (abs(df_withoutna["2014Q1"])>0.85)|
                 (abs(df_withoutna["2014Q2"])>0.85)|
                 (abs(df_withoutna["2014Q3"])>0.85)|
                 (abs(df_withoutna["2014Q4"])>0.85)].index)

ratio2015 = pd.read_csv(path+"2015 portfolio.csv")
code = ratio2015["code"].str.split(".",expand=True)
ratio2015["code"] = code[0]
dict_map = {'technology':0, 'utility':1, 'health care':2, 'consumer discretionary':3,
       'industry':4, 'real estate':5, 'consumer staples':6, 'material':7,
       'energy':8, 'finance':9}
ratio2015["ind_"] = ratio2015["ind_"].map(dict_map)


ratio2015["report_period"] = pd.to_datetime(ratio2015["report_period"],format = "%m/%d/%y")


ratio2012q3 = ratio2015[ratio2015["report_period"]<=pd.Timestamp(2012,9,30)]
ratio2012withoutna = ratio2012q3.dropna(how="any")
data2012q32012q4 = ratio2012withoutna.merge(df_withoutextreme[["code","2012Q4"]],on = "code")
data2012q32012q4["return"] = data2012q32012q4["2012Q4"]
data2012q32012q4=data2012q32012q4.drop(["2012Q4"],axis=1)

ratio2012q4 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2012,12,31)]
ratio2012q4withoutna = ratio2012q4.dropna(how="any")
data2012q42013q1 = ratio2012q4withoutna.merge(df_withoutextreme[["code","2013Q1"]],on = "code")
data2012q42013q1["return"] = data2012q42013q1["2013Q1"]
data2012q42013q1=data2012q42013q1.drop(["2013Q1"],axis=1)

ratio2013q1 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2013,3,31)]
ratio2013q1withoutna = ratio2013q1.dropna(how="any")
data2013q12013q2 = ratio2013q1withoutna.merge(df_withoutextreme[["code","2013Q2"]],on = "code")
data2013q12013q2["return"] = data2013q12013q2["2013Q2"]
data2013q12013q2=data2013q12013q2.drop(["2013Q2"],axis=1)

ratio2013q2 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2013,6,30)]
ratio2013q2withoutna = ratio2013q2.dropna(how="any")
data2013q22013q3 = ratio2013q2withoutna.merge(df_withoutextreme[["code","2013Q3"]],on = "code")
data2013q22013q3["return"] = data2013q22013q3["2013Q3"]
data2013q22013q3=data2013q22013q3.drop(["2013Q3"],axis=1)

ratio2013q3 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2013,9,30)]
ratio2013q3withoutna = ratio2013q3.dropna(how="any")
data2013q32013q4 = ratio2013q3withoutna.merge(df_withoutextreme[["code","2013Q4"]],on = "code")
data2013q32013q4["return"] = data2013q32013q4["2013Q4"]
data2013q32013q4=data2013q32013q4.drop(["2013Q4"],axis=1)

ratio2013q4 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2013,12,31)]
ratio2013q4withoutna = ratio2013q4.dropna(how="any")
data2013q42014q1 = ratio2013q4withoutna.merge(df_withoutextreme[["code","2014Q1"]],on = "code")
data2013q42014q1["return"] = data2013q42014q1["2014Q1"]
data2013q42014q1=data2013q42014q1.drop(["2014Q1"],axis=1)

ratio2014q1 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2014,3,31)]
ratio2014q1withoutna = ratio2014q1.dropna(how="any")
data2014q12014q2 = ratio2014q1withoutna.merge(df_withoutextreme[["code","2014Q2"]],on = "code")
data2014q12014q2["return"] = data2014q12014q2["2014Q2"]
data2014q12014q2=data2014q12014q2.drop(["2014Q2"],axis=1)

ratio2014q2 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2014,6,30)]
ratio2014q2withoutna = ratio2014q2.dropna(how="any")
data2014q22014q3 = ratio2014q2withoutna.merge(df_withoutextreme[["code","2014Q3"]],on = "code")
data2014q22014q3["return"] = data2014q22014q3["2014Q3"]
data2014q22014q3=data2014q22014q3.drop(["2014Q3"],axis=1)

ratio2014q3 = ratio2015[ratio2015["report_period"]==pd.Timestamp(2014,9,30)]
ratio2014q3withoutna = ratio2014q3.dropna(how="any")
data2014q32014q4 = ratio2014q3withoutna.merge(df_withoutextreme[["code","2014Q4"]],on = "code")
data2014q32014q4["return"] = data2014q32014q4["2014Q4"]
data2014q32014q4=data2014q32014q4.drop(["2014Q4"],axis=1)

data_for_regression_20122015 = pd.concat([data2012q32012q4,data2012q42013q1,data2013q12013q2,data2013q22013q3,
                                         data2013q32013q4,data2013q42014q1,data2014q12014q2,data2014q22014q3,data2014q32014q4])
data_for_regression_20122015 = data_for_regression_20122015.drop(["WGSD_COM_EQ","MKT_CAP_ARD","report_period"],axis=1)


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

x = data_for_regression_20122015.iloc[:,1:14]
y = data_for_regression_20122015["return"]
s = StandardScaler()
x_normalized = s.fit_transform(x)

x_normalized = pd.DataFrame(x_normalized,columns=x.columns)


y = y.reset_index(drop=True)
X = sm.add_constant(x_normalized)
mod = sm.OLS(y,X)
fii = mod.fit()
fii.summary()

# 按照model筛选
ratio2015_predict = ratio2015[ratio2015["report_period"]==pd.Timestamp(2014,9,30)].drop(
    ["WGSD_COM_EQ","MKT_CAP_ARD","report_period"],axis=1).dropna(how="any")
code = ratio2015_predict["code"].reset_index().drop(["index"],axis=1)
x_test = ratio2015_predict.iloc[:,1:14]
s = StandardScaler()
x_test_norm = s.fit_transform(x_test)
x_test_norm = pd.DataFrame(x_test_norm,columns=x_test.columns)

x_test_norm = sm.add_constant(x_test_norm)
x_test_norm
y = fii.predict(x_test_norm)

y_df = pd.concat((code,y),axis=1)
y_df = y_df.rename(columns={0:"expected_return"})
long = y_df.sort_values(["expected_return"],ascending=False).iloc[:50,]
short = y_df.sort_values(["expected_return"],ascending=True).iloc[:50,]

fullreturn =  pd.read_excel(path+"2015fullreturn.xlsx")

fullreturn["code"] = fullreturn["code"].astype(str).str.pad(6,fillchar="0")
longreturn = fullreturn[fullreturn["code"].isin(long["code"])]
longreturn = longreturn.dropna(how="any")
longreturn = longreturn.loc[(longreturn!=0).all(1)]

daily_return = []

for i in range(1,longreturn.shape[1]-1):
    dailyreturn = (2*(longreturn.iloc[:,i+1]-longreturn.iloc[:,i])/longreturn.iloc[:,i]).mean()
    daily_return.append(dailyreturn)
    
shortreturn = fullreturn[fullreturn["code"].isin(short["code"])]
shortreturn = shortreturn.dropna(how="any")
shortreturn = shortreturn.loc[(shortreturn!=0).all(1)]

short_daily_return = []

for i in range(1,shortreturn.shape[1]-1):
    dailyreturn = (-1*(shortreturn.iloc[:,i+1]-shortreturn.iloc[:,i])/shortreturn.iloc[:,i]).mean()
    short_daily_return.append(dailyreturn)

final_day_return = np.array([1+(daily_return[i]+short_daily_return[i])/2 for i in range(len(daily_return))])
final_day_return_ = np.cumprod(final_day_return)

final_day_return
import matplotlib.pyplot as plt

sz = pd.read_csv(path+"2015SZreturn.csv")
sz_return = sz["return"]
sz_return = sz_return.to_list()
sz_return_float = np.array([1+float(item[:-1])/100 for item in sz_return])
sz_return_float_ = sz_return_float.cumprod()

plt.figure()
plt.title("1 year return compared to SZ Index")
plt.xlabel("day")
plt.ylabel("performance")
plt.plot(final_day_return_,label = "my portfolio")
plt.plot(sz_return_float_,label="SZ index performance")
plt.legend()
plt.show()

sz_return_float
from scipy.stats import kurtosis, skew
kurtosis(final_day_return)
