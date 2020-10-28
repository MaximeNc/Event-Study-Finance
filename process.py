import pandas as pd
import numpy as np
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

def fctCAAR(df):
    """
    The fonction returns the dataframe for the final AAR and CAAR
    It takes as argument a list of events to aggregate (Evt1 + Evt2 + ... )
    """
    #Creates empty dataframe for final AAR and CAAR
    dfCAAR = pd.DataFrame([0 for i in range(-20,21)], columns =["AAR"])
    dfCAAR.index = [i for i in range(-20,21)]
    dfCAAR["CAAR"] = 0
    dfCAAR["S(AAR)"] = 0
    dfCAAR["S(CAAR)"] = 0

    dfAR = pd.DataFrame([0 for i in range(0,41)], columns =["AR-evt-1"])
    dfAR.index = [i for i in range(-20,21)]
    dfCAR = pd.DataFrame([0 for i in range(0,41)], columns =["CAR-evt-1"])
    dfCAR.index = [i for i in range(-20,21)]
    

    nb_evt = 0 # initializing number of events to 0

    print("Number of events = ", df["evt"].sum())

    #Loop each time there is an event in the dataframe
    for eventdate in df.loc[df.evt==1,].index.values:
        nb_evt = nb_evt + 1 # +1 evt
        t0 = df.index.get_loc(eventdate)
        t1 = t0 - 250
        t2 = t0 - 20
        t3 = t0 + 20   
        
        #Estimate the market model with a regression between -250 and -20
        y = np.array(df["Rt"].iloc[t1:t2])
        x = np.array(df["Rm"].iloc[t1:t2])
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        alph = results.params[0]
        beta = results.params[1]

        #creates a dataframe that contains AR and CAR (the diff between the model and the actual returns from -20 to +20)
        df_window = df[["Rt", "Rm"]].iloc[t2:t3+1]
        df_window.index = [i for i in range(-20,21)]
        df_window["AR"] = df_window["Rt"] - (alph + beta*df_window["Rm"])
        df_window["CAR"] = df_window["AR"].cumsum()

        dfCAAR["AAR"] = dfCAAR["AAR"] + df_window["AR"]
        dfCAAR["CAAR"] = dfCAAR["CAAR"] + df_window["CAR"]

        ColNameAR = "AR-evt-%s" % nb_evt
        ColNameCAR = "CAR-evt-%s" % nb_evt
        dfAR[str(ColNameAR)] = df_window["AR"]
        dfCAR[str(ColNameCAR)] = df_window["CAR"]

    dfCAAR["AAR"]= dfCAAR["AAR"] / nb_evt
    dfCAAR["CAAR"]= dfCAAR["CAAR"] / nb_evt
    dfCAAR["S(AAR)"] = dfAR.std(axis=1)
    dfCAAR["S(CAAR)"] = dfCAR.std(axis=1)

    dfCAAR["t-stat(AAR)"] = dfCAAR["AAR"] / (dfCAAR["S(AAR)"] / np.sqrt(nb_evt))
    dfCAAR["t-stat(CAAR)"] = dfCAAR["CAAR"] / (dfCAAR["S(CAAR)"] / np.sqrt(nb_evt))
    return dfCAAR



filename = "data/ex.csv"
df = pd.read_csv(filename)
df = df.set_index(pd.DatetimeIndex(df["dates"]))
df = df.drop(["dates"], axis=1)
#Compute log returns
df["Rt"] = np.log(df["close"]) - np.log(df["close"].shift(1))
df["Rm"] = np.log(df["index"]) - np.log(df["index"].shift(1))
#Compute normal returns
#df["Rt"] = (df["Close"] / df["Close"].shift(1) ) - 1
#df["Rm"] = (df["Mkt"] / df["Mkt"].shift(1) ) - 1
df = df.dropna()

dfCAAR_Loss = fctCAAR(df)
print(dfCAAR_Loss)

fig, ax = plt.subplots()
ax.plot(dfCAAR_Loss["CAAR"])
plt.xticks(np.arange(-20, 25, 5))
plt.axvline(x=0,color="red")
#plt.savefig('plot1.png')
ax.grid()
plt.show()

