# Computing resource price study

Price study for various computing resources.

# Cloud resources

![cloud](cloud.png)

Based on an analysis of EC2 price data, the correlation of price, vCPUs, memory, storage, and network bandwidth correlates as shows in the figure above.

OLS regression gives:

```
OLS Regression Results:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:           Price_Dollar   R-squared:                       0.877
Model:                            OLS   Adj. R-squared:                  0.875
Method:                 Least Squares   F-statistic:                     409.3
Date:                Thu, 13 Feb 2025   Prob (F-statistic):          2.62e-103
Time:                        16:06:51   Log-Likelihood:                -183.63
No. Observations:                 235   AIC:                             377.3
Df Residuals:                     230   BIC:                             394.6
Df Model:                           4                                         
Covariance Type:            nonrobust                                         
================================================================================
                   coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------
const           -0.1438      0.085     -1.685      0.093      -0.312       0.024
vCPUs_Num        0.0305      0.002     13.554      0.000       0.026       0.035
Memory_GiB       0.0049      0.000     13.099      0.000       0.004       0.006
Storage_GB    4.115e-05   2.34e-05      1.760      0.080   -4.92e-06    8.72e-05
Network_Gbit     0.0192      0.005      3.719      0.000       0.009       0.029
==============================================================================
Omnibus:                      226.033   Durbin-Watson:                   1.331
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             5942.358
Skew:                           3.778   Prob(JB):                         0.00
Kurtosis:                      26.448   Cond. No.                     6.01e+03
==============================================================================
```

Resulting in a price of:

- $0.0305 per hour per vCPU
- $0.0049 per hour per GiB of memory
- $4.115e-05 per hour per GB of storage
- $0.0192 per hour per Gbit of network bandwidth
- With an intercept of: -0.1438

Hence:

_-0.1438 + 0.0305 * 2 + 0.0049 * 8 + 4.115e-05 * 118 + 0.0192 * 12.5_

Gives an approximate value of:

_$0.20125569999999998 per hour_

For:

| Instance   | Memory  | vCPUS   |Storage         | Bandwidth          | Price |
|------------|---------|---------|----------------|--------------------|----------------|
| m6id.large | 8.0 GiB | 2 vCPUs |118 GB NVMe SSD | Up to 12.5 Gigabit | $0.1187 hourly |

For an error of:

_$0.08255569999999998_