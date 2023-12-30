import pyomo.environ as pyo
import pandas as pd
from statistics import NormalDist
import matplotlib.pyplot as plt

###################################Heat price data###################################
HPrice_energy = [0.521, 0.521, 0.521, 0.359, 0.164, 0.100, 0.100, 0.100, 0.145, 0.359, 0.414, 0.521]  # SEK/kWh
# HPrice_effect = {'f':{9510, 14560, 27863}, 'v':{911, 861, 808}} #f: SEK/year, v: #SEK/kWh year, 0-100, 101-250, 251-500
HPrice_effect = {'f': [9510], 'v': [911]}  # f: SEK/year, v: #SEK/kWh year for 0-100 kWh thermal
# HPrice_Efficiency = 0.007 #SEK/kWh Degree of celsius for (Oct-April only)

Subscription_fee = 147.5 / 30  # Subscription fee SEK/day
Transmission_fee = 0.255  # Electricity transmission fee SEK/kWh
Tax_fee = 0.49  # Tax fee SEK/kWh
Effect_fee = 36.25 / 30  # Effect fee SEK/kW
Incentive_fee = 0.08  # SEK/kWh due to loss reduction
Taxreduction_fee = 0.6  # SEK/kWh tax reduction from tax agency
############################Data preparation#################################
# optimization in this iteration will be done for time interval [StartTimeControl,EndTimeControl]
Temperature = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 14, 14, 14, 13, 13, 12, 13, 13, 13, 13, 13, 12]

# Spot_market data
spot_price = [0.221990000000000, 0.228250000000000, 0.218800000000000,
              0.207140000000000, 0.200660000000000, 0.193510000000000,
              0.208570000000000, 0.202200000000000, 0.220230000000000,
              0.235400000000000, 0.241890000000000, 0.284770000000000,
              0.507080000000000, 0.549960000000000, 0.439800000000000,
              0.253430000000000, 0.545680000000000, 1.04067000000000,
              1.48508000000000, 0.271790000000000, 0.212750000000000,
              0.205930000000000, 0.182300000000000, 0.159320000000000]

# Flex price
# flex_price = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               sum(spot_price)/24, sum(spot_price)/24, sum(spot_price)/24,
#               sum(spot_price)/24, sum(spot_price)/24, sum(spot_price)/24,
#               sum(spot_price)/24, sum(spot_price)/24, sum(spot_price)/24,
#               0, 0, 0, 0]


flex_price = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              max(spot_price), max(spot_price), max(spot_price),
              max(spot_price), max(spot_price), max(spot_price),
              max(spot_price), max(spot_price), max(spot_price),
              0, 0, 0, 0]

# flex_price = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               spot_price[11], spot_price[12], spot_price[13],
#               spot_price[14], spot_price[15], spot_price[16],
#               spot_price[17], spot_price[18], spot_price[19],
#               0, 0, 0, 0]


# Electrical load data
HSBelecLoad = [9.63000000000000, 10.4900000000000, 9.14000000000000, 8.40000000000000,
               9.37000000000000, 8.12000000000000, 9.40000000000000, 12.0800000000000,
               14.3000000000000, 15.1500000000000, 16.6600000000000, 16.1300000000000,
               15.3000000000000, 11.8500000000000, 11.9800000000000, 14.2300000000000,
               15.7900000000000, 17.9800000000000, 15.0500000000000, 16.0800000000000,
               13.8700000000000, 11.5900000000000, 8.89000000000000, 10.7100000000000]

# Heat load data
HSBheatLoad = [12.4803448300000, 12.2123333300000, 11.8865517200000, 11.5936666700000,
               11.3150000000000, 11.0182608700000, 10.7440740700000, 10.3885714300000,
               9.80900000000000, 9.74689655200000, 10.1250000000000, 10.5923333300000,
               11.2810344800000, 11.8136666700000, 12.1310000000000, 12.4350000000000,
               12.6665517200000, 12.9186666700000, 14.1130000000000, 14.0800000000000,
               13.4406666700000, 12.8168965500000, 12.3282758600000, 11.9440000000000]

# PV Generation data
HSBpvgen = [0, 0, 0, 0, 0, 0, 0.0600000000000000, 0.630000000000000, 2.02000000000000, 4.67000000000000,
            4.23000000000000, 7.35000000000000, 5.18000000000000, 2.67000000000000, 1.01000000000000,
            0.170000000000000, 0.0100000000000000, 0, 0, 0, 0, 0, 0, 0]

# COP of HPs
VP1_GT1_PV = 35  # Set point of HP1
pHPmax = 5  # HP's maximum electricity limit
hHPmax = 15  # HP's maximum heat limit

# Technical data of BES
pBESmax = 3  # BE's maximum charger limit power
eBESmax = 7.2  # BE's battery capacity
SOCmax = 0.90  # Maximum SOC of BES
SOCmin = 0.10  # Minimum SOC of BES
SOCini = 0.90  # Initial SOC of BES
eff = 0.923  # BE's charging/discharging efficiency

# Technical data of electricity and heat grids
pGridmax = 50  # Grid's maximum heat limit
hDHmax = 30  # DH's maximum heat limit
#############################Optimization model###################################
model = pyo.ConcreteModel()
# Sets
model.T = pyo.Set(initialize=range(0, 24))

# Variables
model.hDH = pyo.Var(model.T, bounds=(0, hDHmax), within=pyo.NonNegativeReals, initialize=0)
model.hHP = pyo.Var(model.T, bounds=(0, hHPmax), within=pyo.NonNegativeReals, initialize=0)
model.pHP = pyo.Var(model.T, bounds=(0, pHPmax), within=pyo.NonNegativeReals, initialize=0)
model.HP_COP = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.pCH = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.pDC = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.uCH = pyo.Var(model.T, within=pyo.Binary, initialize=0)
model.uDC = pyo.Var(model.T, within=pyo.Binary, initialize=0)
model.SOC = pyo.Var(model.T, bounds=(SOCmin, SOCmax), within=pyo.NonNegativeReals, initialize=SOCini)
model.pGridim = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.pGridex = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)
model.uGridim = pyo.Var(model.T, within=pyo.Binary, initialize=0)
model.uGridex = pyo.Var(model.T, within=pyo.Binary, initialize=0)
model.heatpeak = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
model.elecpeak = pyo.Var(within=pyo.NonNegativeReals, initialize=0)
model.Flex = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)




pGridim0 = [12.126069, 12.932467, 11.517310, 10.718733, 11.633000,
            10.323652, 11.488815, 13.527714, 14.241800, 12.429379,
            14.455000, 10.898467, 12.376207, 9.817895, 13.396200,
            18.571627, 18.313310, 18.571627, 14.872600, 18.571627,
            16.558133, 14.153379, 11.355655, 13.098800]


# Objective
def rule_1(model):
    return (Subscription_fee
            + sum((Transmission_fee + Tax_fee + 1 * spot_price[t]) * model.pGridim[t] for t in model.T)
            - sum((Incentive_fee + Taxreduction_fee + 1 * spot_price[t]) * model.pGridex[t] for t in model.T)
            + model.elecpeak * Effect_fee
            + sum(HPrice_energy[int(12 - 1)] * model.hDH[t] for t in model.T)
            + (HPrice_effect['v'][0] * model.heatpeak) / (365 * 24)
            - sum((1 * flex_price[t] + Transmission_fee + Tax_fee) * model.Flex[t] for t in model.T)
            )


model.obj = pyo.Objective(rule=rule_1, sense=pyo.minimize)


# Constraints
def rule_C1(model, t):
    return model.hHP[t] + model.hDH[t] == HSBheatLoad[t]
model.C1 = pyo.Constraint(model.T, rule=rule_C1)


def rule_C2(model, t):
    return model.pGridim[t] + model.pDC[t] == \
        model.pGridex[t] + model.pHP[t] + model.pCH[t] + \
        (HSBelecLoad[t] - HSBpvgen[t])
model.C2 = pyo.Constraint(model.T, rule=rule_C2)


def rule_C3(model, t):
    return model.hHP[t] <= model.pHP[t] * 5
model.C3 = pyo.Constraint(model.T, rule=rule_C3)


def rule_C4(model, t):
    return model.heatpeak >= model.hDH[t]
model.C4 = pyo.Constraint(model.T, rule=rule_C4)


def rule_C5(model, t):
    return model.elecpeak >= model.pGridim[t]
model.C5 = pyo.Constraint(model.T, rule=rule_C5)


def rule_C6(model, t):
    return model.pCH[t] <= pBESmax * model.uCH[t]
model.C6 = pyo.Constraint(model.T, rule=rule_C6)


def rule_C7(model, t):
    return model.pDC[t] <= pBESmax * model.uDC[t]
model.C7 = pyo.Constraint(model.T, rule=rule_C7)


def rule_C8(model, t):
    return model.uCH[t] + model.uDC[t] <= 1
model.C8 = pyo.Constraint(model.T, rule=rule_C8)


def rule_C9(model, t):
    if t == 0:
        return model.SOC[t] == SOCini + eff * model.pCH[t] / eBESmax - model.pDC[t] / (eff * eBESmax)
    else:
        return model.SOC[t] == model.SOC[t - 1] + eff * model.pCH[t] / eBESmax - model.pDC[t] / (eff * eBESmax)
model.C9 = pyo.Constraint(model.T, rule=rule_C9)


# def rule_C9_2(model):
#         return model.SOC[23] == SOCini
# model.C9_2 = pyo.Constraint(rule=rule_C9_2)

def rule_C10(model, t):
    return model.pGridim[t] <= pGridmax * model.uGridim[t]
model.C10 = pyo.Constraint(model.T, rule=rule_C10)


def rule_C11(model, t):
    return model.pGridex[t] <= pGridmax * model.uGridex[t]
model.C11 = pyo.Constraint(model.T, rule=rule_C11)


def rule_C12(model, t):
    return model.uGridim[t] + model.uGridex[t] <= 1
model.C12 = pyo.Constraint(model.T, rule=rule_C12)


def rule_C13(model, t):
    if flex_price[t] != 0:
        return model.Flex[t] <= (pGridim0[t] - model.pGridim[t] -
                                 NormalDist().inv_cdf(0.99)*((0.05*HSBelecLoad[t])**2 +
                                                             (0.2*HSBpvgen[t]+0.02*max(HSBpvgen[:]))**2)**0.5)
    else:
        return model.Flex[t] <= 0
model.C13 = pyo.Constraint(model.T, rule=rule_C13)


opt = pyo.SolverFactory('gurobi', solver_io="python", executable=r"C:\gurobi1100\win64")
reults = opt.solve(model)

scheduling = pd.DataFrame([pyo.value(model.pGridim[:]), pyo.value(model.pGridex[:]),
                           pyo.value(model.pCH[:]), pyo.value(model.pDC[:]),
                           pyo.value(model.pHP[:]), HSBelecLoad[:], HSBpvgen[:]

                           ], index=['Pgrid_imp', 'Pgrid_exp',
                                     'PBES_CH', 'PBES_DC',
                                     'PHP', 'Load', 'PV'])
scheduling.loc['Pgrid_imp', :] = scheduling.loc['Pgrid_imp', :].apply(lambda x: -x)
scheduling.loc['PV', :] = scheduling.loc['PV', :].apply(lambda x: -x)
scheduling.loc['PBES_DC', :] = scheduling.loc['PBES_DC', :].apply(lambda x: -x)
# Separate positive and negative values for Pgrid_imp and PBES_DC
# Transpose the DataFrame to have components as columns and hours as rows
scheduling_transposed = scheduling.T

# Create a stacked bar chart
scheduling_transposed.plot(kind='bar', stacked=True)

plt.xlabel('Hour of the Day')
plt.ylabel('Power (kW)')
plt.legend()
plt.show()