
# coding: utf-8
'''

# Stock Market Analysis:

1.) What was the change in price of the stock over time?
2.) What was the daily return of the stock on average?
3.) What was the moving average of the various stocks?
4.) What was the correlation between different stocks' closing prices?
4.) What was the correlation between different stocks' daily returns?
5.) How much value do we put at risk by investing in a particular stock?
6.) How can we attempt to predict future stock behavior?

'''
# In[2]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')

#NOTE

All the depricated data scaping methods are mentioned below

# In[16]:


'''

from pandas_datareader import wb, data      #DEPRICATED
depimport pandas_datareader.data as pdr     #DEPRICATED

from pandas.io.data import DataReader       #DEPRICATED

'''


# In[17]:


from datetime import datetime


# In[22]:


end = datetime.now()
start = datetime(end.year - 1,end.month,end.day)


# In[23]:


#Getting DATA from Yahoo/Google

tech_list = ['AAPL','GOOG','MSFT','AMZN']


# In[24]:


from yahoo_fin import stock_info as si


# In[25]:


# globals() used for setting all the string names like AAPL, GOOG as global variables

for stock in tech_list:
    globals()[stock] = si.get_data(stock, start, end)


# In[26]:


# This global variable can get the data for this particular Stock

AAPL.head()


# In[27]:


MSFT.head()


# In[28]:


#Opening price, closing price, low, high and Split stock changes

GOOG.head()


# In[29]:


#describes all the statistical data for the stock data.

AAPL.describe()


# In[30]:


#taotal count and other info

AAPL.info()


# In[31]:


#Set Figure size not necessary

AAPL['adjclose'].plot(legend=True, figsize=(15,6))


# In[32]:


AAPL['volume'].plot(legend=True, figsize=(15,6))


# In[39]:


# For financial data Moving averages is one of the important factors for analyzing
# and predicting stock prices and other entities.


ma_day = [10, 20, 50]

'''
for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))                     # DEPRICATED
    AAPL[column_name] = pd.rolling_mean(AAPL['adjclose'], ma)     # DEPRICATED
    
'''    


# In[40]:


#moving average and ints convergence of foloow
#This can give different treands in data

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    AAPL[column_name] = AAPL['adjclose'].rolling(ma).mean()


# In[42]:


#Plotting the data for All categories

AAPL[['adjclose', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False,
                                                                               figsize=(15,6))


# In[68]:


#Creating Daily Return using Percentage change in values

AAPL['daily return'] = AAPL['adjclose'].pct_change()


# In[44]:


#Risk Analysis using plotting daily returns (ups and downs)

AAPL['daily return'].plot(figsize=(15,6), legend=True, linestyle='--', marker='o')


# In[45]:


#Dive deep in seaborn #DEPRICATED normed

sns.distplot(AAPL['daily return'].dropna(), bins=100, color='purple')


# In[46]:


#Pandas histogram for smae implementation

AAPL['daily return'].hist(bins=100)


# In[47]:


all_data = pd.DataFrame()

all_data = [AAPL.adjclose , GOOG.adjclose, MSFT.adjclose, AMZN.adjclose]
total = pd.DataFrame(all_data, index=['AAPL' , 'GOOG', 'MSFT', 'AMZN']).transpose()


all_close = [AAPL.close , GOOG.close, MSFT.close, AMZN.close]
total_close = pd.DataFrame(all_close, index=['AAPL' , 'GOOG', 'MSFT', 'AMZN']).transpose()


# In[72]:


tech_rets = total.pct_change()
tech_rets.head()


# In[48]:


total.head()


# In[73]:


#pearson value = 1 (Pearson product-moment correlation coefficient)

sns.jointplot('GOOG','GOOG', tech_rets, kind='scatter', color='seagreen' )


# In[74]:


#pearson value = 0.66 (Pearson product-moment correlation coefficient)

sns.jointplot('GOOG', 'MSFT', tech_rets, kind='scatter')


# In[52]:


total.head()


# In[75]:


sns.pairplot(tech_rets.dropna(), kind='scatter')


# In[76]:


return_fig = sns.PairGrid(tech_rets.dropna())

return_fig.map_upper(plt.scatter, color='purple')
return_fig.map_lower(sns.kdeplot, cmap='cool_d')
return_fig.map_diag(plt.hist)


# In[78]:


return_fig = sns.PairGrid(total.dropna())

return_fig.map_upper(plt.scatter, color='purple')
return_fig.map_lower(sns.kdeplot, cmap='cool_d')
return_fig.map_diag(plt.hist, bins=30)


# In[ ]:


#sns.corrplot(total.dropna(),annot=True)   #DEPRICATED

#Seaborn Correlation plot is depricated, instead heatmaps are more robust to outliers
# and gives better picture than correlation matrices


# In[79]:


corr = tech_rets.corr()

# @This masking function is taken from internet source 
mask = np.zeros_like(corr, dtype=np.bool)
# @This masking function is taken from internet source
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, annot=True, mask=mask)


# # RISK ANALYSYS
# 

# In[90]:


rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)

# Set the x and y limits of the plot (optional, remove this if you don't see anything in your plot)
plt.ylim([0.01,0.030])
plt.xlim([-0.001,0.003])

#Set the plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

# @Content for annotation extracted from matplotlib online resourse
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.2'))


# https://matplotlib.org/users/annotations_guide.html
# Setting annotations


# In[66]:


#Value at risk

#BOOTSTRAP METHOD: quantile and percentile

sns.distplot(AAPL['daily return'].dropna(), bins=100, color='purple')


# In[92]:


# Worst case loss with 95% confidence you can loose 0.02795, it cannot exceed this amount

rets['AAPL'].quantile(0.05)


# # MONTE CARLO METHOD STOCK SIMULATION
Formula for calcullations : ΔS=S(μΔt+σϵ√Δt)

Here the S (stock price is multiplied by 2 terms)
1st term: Drift :

    Drift is forward movement of values based on average daily return multiplied by the change of time.
    
2nd term: Shock :
    This is a vertical movement of price (up or down) randomly.
   
For predicting stock everytime the stock price will Drift and experience a SHock either up or down. Multiple simulations of these will generate a histogram of lines that will predict stock price at a certain point in time.
# In[97]:


days= 365

dt = 1/days

#Average return
mu = rets.mean()['GOOG']

#Standard deviation on daily return
sigma = rets.std()['GOOG']


# In[98]:


def stock_monte_carlo(start_price, days, mu, sigma):
    
    price = np.zeros(days)
    price[0] = start_price
    
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in range(1, days):
        
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        
        drift[x] = mu*dt
        
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


# In[95]:


GOOG.head()


# In[99]:


start_price = 1035.50

for run in range(100):
    plt.plot(stock_monte_carlo(start_price, days, mu, sigma))
    
    
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Monte Carlo Analysis for GOOGLE')


# In[100]:


runs = 10000

sims = np.zeros(runs)

for run in range(runs):
    sims[run] = stock_monte_carlo(start_price, days, mu, sigma)[days-1]


# In[102]:


#PLot a histoogram

q = np.percentile(sims, 1)

plt.hist(sims, bins=200)


plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % sims.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');

