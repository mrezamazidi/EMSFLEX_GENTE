from EMSFLEX_Robust import EMSFLEXfunc
from EMSEnergy import EMSEnergyfunc
from EMSFLEX_Provide import EMSFLEX_providefunc
import pandas as pd

# Spot_market data
spot_price = [0.221990000000000, 0.228250000000000, 0.218800000000000,
              0.207140000000000, 0.200660000000000, 0.193510000000000,
              0.208570000000000, 0.202200000000000, 0.220230000000000,
              0.235400000000000, 0.241890000000000, 0.284770000000000,
              0.507080000000000, 0.549960000000000, 0.439800000000000,
              0.253430000000000, 0.545680000000000, 1.04067000000000,
              1.48508000000000, 0.271790000000000, 0.212750000000000,
              0.205930000000000, 0.182300000000000, 0.159320000000000]

Flex_price = [sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24, sum(spot_price) / 24,
              sum(spot_price) / 24, sum(spot_price) / 24
              ]
Flex_price = [Flex_price[i]*10 for i in range(len(Flex_price))]

HSBelecloadHistData = [9.63000000000000, 10.4900000000000, 9.14000000000000, 8.40000000000000,
                       9.37000000000000, 8.12000000000000, 9.40000000000000, 12.0800000000000,
                       14.3000000000000, 15.1500000000000, 16.6600000000000, 16.1300000000000,
                       15.3000000000000, 11.8500000000000, 11.9800000000000, 14.2300000000000,
                       15.7900000000000, 17.9800000000000, 15.0500000000000, 16.0800000000000,
                       13.8700000000000, 11.5900000000000, 8.89000000000000, 10.7100000000000,
                       9.63000000000000, 10.4900000000000, 9.14000000000000, 8.40000000000000,
                       9.37000000000000, 8.12000000000000, 9.40000000000000, 12.0800000000000,
                       14.3000000000000, 15.1500000000000, 16.6600000000000, 16.1300000000000,
                       15.3000000000000, 11.8500000000000, 11.9800000000000, 14.2300000000000,
                       15.7900000000000, 17.9800000000000, 15.0500000000000, 16.0800000000000,
                       13.8700000000000, 11.5900000000000, 8.89000000000000, 10.7100000000000
                       ]
HSBheatloadHistData = [12.4803448300000, 12.2123333300000, 11.8865517200000, 11.5936666700000,
                       11.3150000000000, 11.0182608700000, 10.7440740700000, 10.3885714300000,
                       9.80900000000000, 9.74689655200000, 10.1250000000000, 10.5923333300000,
                       11.2810344800000, 11.8136666700000, 12.1310000000000, 12.4350000000000,
                       12.6665517200000, 12.9186666700000, 14.1130000000000, 14.0800000000000,
                       13.4406666700000, 12.8168965500000, 12.3282758600000, 11.9440000000000,
                       12.4803448300000, 12.2123333300000, 11.8865517200000, 11.5936666700000,
                       11.3150000000000, 11.0182608700000, 10.7440740700000, 10.3885714300000,
                       9.80900000000000, 9.74689655200000, 10.1250000000000, 10.5923333300000,
                       11.2810344800000, 11.8136666700000, 12.1310000000000, 12.4350000000000,
                       12.6665517200000, 12.9186666700000, 14.1130000000000, 14.0800000000000,
                       13.4406666700000, 12.8168965500000, 12.3282758600000, 11.9440000000000
                       ]

HSBpvgenHistData = [0, 0, 0, 0, 0, 0, 0.0600000000000000, 0.630000000000000, 2.02000000000000, 4.67000000000000,
                    4.23000000000000, 7.35000000000000, 5.18000000000000, 2.67000000000000, 1.01000000000000,
                    0.170000000000000, 0.0100000000000000, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0.0600000000000000, 0.630000000000000, 2.02000000000000, 4.67000000000000,
                    4.23000000000000, 7.35000000000000, 5.18000000000000, 2.67000000000000, 1.01000000000000,
                    0.170000000000000, 0.0100000000000000, 0, 0, 0, 0, 0, 0, 0
                    ]

Spot_price = [0.221990000000000, 0.228250000000000, 0.218800000000000,
              0.207140000000000, 0.200660000000000, 0.193510000000000,
              0.208570000000000, 0.202200000000000, 0.220230000000000,
              0.235400000000000, 0.241890000000000, 0.284770000000000,
              0.507080000000000, 0.549960000000000, 0.439800000000000,
              0.253430000000000, 0.545680000000000, 1.04067000000000,
              1.48508000000000, 0.271790000000000, 0.212750000000000,
              0.205930000000000, 0.182300000000000, 0.159320000000000,
              0.221990000000000, 0.228250000000000, 0.218800000000000,
              0.207140000000000, 0.200660000000000, 0.193510000000000,
              0.208570000000000, 0.202200000000000, 0.220230000000000,
              0.235400000000000, 0.241890000000000, 0.284770000000000,
              0.507080000000000, 0.549960000000000, 0.439800000000000,
              0.253430000000000, 0.545680000000000, 1.04067000000000,
              1.48508000000000, 0.271790000000000, 0.212750000000000,
              0.205930000000000, 0.182300000000000, 0.159320000000000
              ]

dispatch = {}
Pgrid_imp0 = []
flex_value = []
Alpha_value = 0.95  # Chance constraint parameter
CC_flag = 1  # Chance constraint flag
RL_flag = 0  # Random load flag


for t in range(24):
    HSBelecloadPredict = HSBelecloadHistData[t:t + 24]
    HSBheatloadPredict = HSBheatloadHistData[t:t + 24]
    HSBpvgenPredict = HSBpvgenHistData[t:t + 24]
    Spot_pricePredict = Spot_price[t:t + 24]
    Flex_pricePredict = Flex_price[t:t + 24]

    # output = EMSEnergyfunc(HSBelecLoad=HSBelecloadPredict, HSBheatLoad=HSBheatloadPredict,
    # HSBpvgen=HSBpvgenPredict, spot_price=Spot_pricePredict)

    output = EMSEnergyfunc(HSBelecLoad=HSBelecloadPredict, HSBheatLoad=HSBheatloadPredict, HSBpvgen=HSBpvgenPredict,
                           spot_price=Spot_pricePredict)
    if t == 7:
        Pgrid_imp0 = list(output.iloc[0, :])
        output = EMSFLEXfunc(HSBelecLoad=HSBelecloadPredict, HSBheatLoad=HSBheatloadPredict, HSBpvgen=HSBpvgenPredict,
                             spot_price=Spot_pricePredict, flex_price=Flex_pricePredict, pGridim0=Pgrid_imp0,
                             Flex_req_time=t, Flex_req_period=[11, 19], CC_flag=CC_flag, Alpha_value=Alpha_value)
        Flex0 = list(output.iloc[-1, :].T)
        flex_value = Flex0
        print(flex_value)

    if 7 < t < 20:
        Pgrid_imp0.pop(0)
        Pgrid_imp0.insert(-1, 0)
        flex_value.pop(0)
        flex_value.insert(-1, 0)

    if 11 <= t <= 19:
        output = EMSFLEX_providefunc(HSBelecLoad=HSBelecloadPredict, HSBheatLoad=HSBheatloadPredict,
                                     HSBpvgen=HSBpvgenPredict,
                                     spot_price=Spot_pricePredict, pGridim0=Pgrid_imp0, Flex=flex_value,
                                     RL_flag=RL_flag)

    dispatch[t] = pd.DataFrame(output.iloc[:, 0])

Pimp = [dispatch[i].iloc[0, 0] for i in range(24)]
Pshed = [dispatch[i].iloc[-1, 0] for i in range(11, 20)]
