#Out_sample data

from WindPy import w
import pandas as pd
w.start()

wind_ind1=['882001.WI','882007.WI','882010.WI']
wind_ind1_=['能源','金融','公用事业']
wind_ind2=['882002.WI','882003.WI','882004.WI','882005.WI','882006.WI','882008.WI','882011.WI']
wind_ind2_=['材料','工业','可选消费','日常消费','医疗保健','信息技术','房地产']
dl=w.tdays("2012-09-26", "2021-12-31", "Days=Alldays;Period=Q").Data[0]
def query_wind_():
    for i, j in enumerate(dl):
        if j.strftime("%m")=='12' and j.strftime("%Y")>='2014' and j.strftime("%Y")<='2019':
        # if j.strftime("%m")=='12' and j.strftime("%Y")>='2014' and j.strftime("%Y")<'2017':
            print(j)
            l=[]
            dfr=pd.DataFrame()
            for h,k in enumerate(wind_ind2):
                n, components_ = w.wset("sectorconstituent", f"date={j};windcode={k}", usedf=True)
                codelist = ','.join(components_.wind_code.tolist())
                n, df = w.wss(codelist,
                              "mkt_cap_ard",
                              f"tradeDate={j};unit=1", usedf=True)
                # print(df)
                l=l+df.sort_values('MKT_CAP_ARD').tail(100).index.tolist()
                dfr=pd.concat([dfr,pd.DataFrame({'wind_code':df.sort_values('MKT_CAP_ARD').tail(100).index.tolist(),
                                                    'ind_':[wind_ind2_[h]]*100})])

            for h,k in enumerate(wind_ind1):
                n,components_=w.wset("sectorconstituent", f"date={j};windcode={k}",usedf=True)
                codelist=','.join(components_.wind_code.tolist())
                n,df1=w.wss(codelist, "avg_turn_per",f"startDate={dl[i-4]};endDate={j}",usedf=True)
                l=l+df1[df1>1].dropna().index.tolist()
                dfr = pd.concat(
                    [dfr, pd.DataFrame({'wind_code': df1[df1>1].dropna().index.tolist(),
                                        'ind_': [wind_ind1_[h]] * len(df1[df1>1].dropna())})])
            dfr.to_csv(dl[i+4].strftime("%Y")+'_stock_pool.csv')
            codelist1 = ','.join(set(l))
            dfr1=pd.DataFrame()
            for mx in dl[i-9:i+5]:
                m, df = w.wss(codelist1,
                              "sec_name,debttoassets,pe_lyr,ps_lyr,pcf_nflyr,pb_lyr,yoyprofit,roe_basic"
                              ",yoy_or,roic,roa,assetsturn1,wgsd_com_eq,mkt_cap_ard",
                              f"rptDate={mx};tradeDate={mx};unit=1;rptDate={mx};rptType=1;currencyType=", usedf=True)
                # print(df)
                df['report_period'] = mx
                dfr1=pd.concat([dfr1,df])

            dfr1.join(dfr.set_index('wind_code')['ind_']).to_csv(dl[i+4].strftime("%Y")+'_portfolio.csv')






if __name__ == '__main__':
    # tm()
    # turnover_()
    query_wind_()
