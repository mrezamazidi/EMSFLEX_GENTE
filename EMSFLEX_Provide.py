import pyomo.environ as pyo
import pandas as pd
from numpy import random

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

def EMSFLEX_providefunc(HSBelecLoad, HSBheatLoad, HSBpvgen, spot_price, pGridim0, Flex, RL_flag):

    #Randomness of load
    if RL_flag:
        for i in range(24):
            HSBelecLoad[i] = random.normal(loc=HSBelecLoad[i], scale=0.05 * HSBelecLoad[i])
            HSBpvgen[i] = random.normal(loc=HSBpvgen[i], scale=0.2*HSBpvgen[i]+0.02*max(HSBpvgen[:]))

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
    model.pShed = pyo.Var(model.T, within=pyo.NonNegativeReals, initialize=0)


    # Objective
    def rule_1(model):
        return (Subscription_fee
                + sum((Transmission_fee + Tax_fee + 1 * spot_price[t]) * model.pGridim[t] for t in model.T)
                - sum((Incentive_fee + Taxreduction_fee + 1 * spot_price[t]) * model.pGridex[t] for t in model.T)
                + model.elecpeak * Effect_fee
                + sum(HPrice_energy[int(12 - 1)] * model.hDH[t] for t in model.T)
                + (HPrice_effect['v'][0] * model.heatpeak) / (365 * 24)
                + sum(10000 * model.pShed[t] for t in model.T)
                )
    model.obj = pyo.Objective(rule=rule_1, sense=pyo.minimize)


    # Constraints
    def rule_C1(model, t):
        return model.hHP[t] + model.hDH[t] == HSBheatLoad[t]
    model.C1 = pyo.Constraint(model.T, rule=rule_C1)


    def rule_C2(model, t):
        return model.pGridim[t] + model.pDC[t] == \
            model.pGridex[t] + model.pHP[t] + model.pCH[t] + \
            (HSBelecLoad[t] - HSBpvgen[t]) - model.pShed[t]
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
        return model.pGridim[t] - model.pShed[t] <= (pGridim0[t] - Flex[t])
    model.C13 = pyo.Constraint(model.T, rule=rule_C13)


    opt = pyo.SolverFactory('gurobi', solver_io="python", executable=r"C:\gurobi1100\win64")
    reults = opt.solve(model)

    scheduling = pd.DataFrame([pyo.value(model.pGridim[:]), pyo.value(model.pGridex[:]),
                               pyo.value(model.pCH[:]), pyo.value(model.pDC[:]),
                               pyo.value(model.pHP[:]), HSBelecLoad[:], HSBpvgen[:],
                               pyo.value(model.pShed[:])
                               ], index=['Pgrid_imp', 'Pgrid_exp',
                                         'PBES_CH', 'PBES_DC',
                                         'PHP', 'Load', 'PV', 'pShed'])
    # Separate positive and negative values for Pgrid_imp and PBES_DC
    # Transpose the DataFrame to have components as columns and hours as rows

    return scheduling