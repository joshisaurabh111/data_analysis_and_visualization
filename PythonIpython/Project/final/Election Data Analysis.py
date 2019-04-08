
# coding: utf-8
'''

Analysis of Election Data using matlotlib, Seaborn

1.) Who was being polled and what was their party affiliation?
2.) Did the poll results favor Romney or Obama?
3.) How do undecided voters effect the poll?
4.) Can we account for the undecided voters?
5.) How did voter sentiment change over time?
6.) Can we see an effect in the polls from the debates?

'''
# In[7]:


import pandas as pd
from pandas import Series, DataFrame
import numpy as np


# In[8]:


import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


import requests


# In[11]:


#from StringIO import StringIO            #DEPRICATED

from io import StringIO


# In[12]:


# This is the url link for the poll data in csv form
url = "http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv"

# Getting information in text form and aviod error 
source = requests.get(url).text

poll_data = StringIO(source) 


# In[13]:


poll_df = pd.read_csv(poll_data)


# In[14]:


poll_df.head()


# In[15]:


poll_df.info()


# In[16]:


sns.factorplot('Affiliation', data=poll_df, kind='count')


# In[17]:


sns.factorplot('Affiliation', data=poll_df, kind='count', hue='Population')


# In[18]:


avg = pd.DataFrame(poll_df.mean())

avg.drop('Number of Observations', axis=0, inplace=True)
avg.drop('Question Text', axis=0, inplace=True)
avg.drop('Question Iteration', axis=0, inplace=True)


# In[19]:


avg.head()


# In[20]:


std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations', axis=0, inplace=True)
#avg.drop('Question Text', axis=0, inplace=True)
#avg.drop('Question Iteration', axis=0, inplace=True)


# In[21]:


std.head()


# In[22]:


avg.plot(yerr=std, kind='bar', legend=False)


# In[28]:


poll_avg = pd.concat([avg, std], axis=1, sort=True)
poll_avg.drop('Question Text', axis=0, inplace=True)
poll_avg.drop('Question Iteration', axis=0, inplace=True)

poll_avg


# In[29]:


poll_avg.columns = ['Average', 'STD']


# In[30]:


poll_avg


# In[31]:


#Time series analysis

poll_df.head()


# In[34]:


#Scatter plot over the period of time, converging

poll_df.plot(x='End Date', y=['Obama','Romney','Other','Undecided'], linestyle='', marker='o')


# In[35]:


from datetime import datetime


# In[37]:


poll_df['difference'] = (poll_df.Obama - poll_df.Romney) / 100

poll_df.head()


# In[38]:


poll_df = poll_df.groupby(['Start Date'], as_index=False).mean()

poll_df.head()


# In[48]:


poll_df.plot('Start Date', 'difference', figsize=(15,6), marker='o', linestyle='-', legend=True)


# In[42]:


#Circling thruogh till it finds october 2012

row_in = 0
xlim = []

for date in poll_df['Start Date']:
    if date[0:7] == '2012-10':
        xlim.append(row_in)
        row_in += 1
    else:
        row_in += 1
        
print (min(xlim))
print (max(xlim))


# In[50]:


#Just analyzing October data

poll_df.plot('Start Date', 'difference', figsize=(15,6), marker='o',
             linestyle='-', xlim=(325,352))

#vertical line through the axis

#Certain Debates Occured on these dates and their effects on polling
#Polling results are not ideal

# Oct 3rd
plt.axvline(x=325+2, linewidth=4, color='grey')
# Oct 11th 
plt.axvline(x=325+10, linewidth=4, color='grey')
# Oct 22nd
plt.axvline(x=325+21, linewidth=4, color='grey')


# # Donor Data Set
1.) How much was donated and what was the average donation?
2.) How did the donations differ between candidates?
3.) How did the donations differ between Democrats and Republicans?
4.) What were the demographics of the donors?
5.) Is there a pattern to donation amounts?
# In[51]:


donor_df = pd.read_csv('Election_Donor_Data.csv')


# In[52]:


donor_df.head()


# In[53]:


#Million rows data set (Big Data)

donor_df.info()


# In[54]:


type(donor_df.contbr_zip)


# In[60]:


donor_df['contb_receipt_amt'].value_counts()


# In[62]:


don_mean = donor_df['contb_receipt_amt'].mean()

don_std = donor_df['contb_receipt_amt'].std()


# In[63]:


print ('Donation was %.2f with std %.2f' %(don_mean, don_std))


# In[78]:


#Huge standard Deviation with respect to average


top_donor = donor_df['contb_receipt_amt'].copy()

top_donor.sort_values()

top_donor.head()


# In[79]:


#Getting rid of Negatives (Refunds)

top_donor = top_donor[top_donor > 0]

top_donor.sort_values()

top_donor.value_counts().head(10)


# In[82]:


common_don = top_donor[top_donor < 2500]

common_don.hist(bins=100, figsize=(15,6))


# In[84]:


#Seperating Donations by party, creating new party column

candidate = donor_df.cand_nm.unique()
candidate


# In[91]:


# Seperating Obama because others are republic candidates


# Fast and ditty way to do this

party_map = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

# Now map the party with candidate
donor_df['Party'] = donor_df.cand_nm.map(party_map)


# In[92]:


# Slow

'''
for i in range(0,len(donor_df)):
    if donor_df['cand_nm'].iloc == 'Obama,Barack':
        donor_df['party'].iloc = 'Democrat'
    else:
        donor_df['party'].iloc = 'Republican'
        
'''


# In[95]:


donor_df = donor_df[donor_df.contb_receipt_amt > 0]

donor_df.head(10)


# In[97]:


# Number of Daoaitons for each party 

donor_df.groupby('cand_nm')['contb_receipt_amt'].count()


# In[98]:


# Total dollar amounts for each party

donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()


# In[104]:


cnad_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

i=0

for don in cnad_amount:
    print ('The Dandidate %s raised %.0f dollars' %(cnad_amount.index[i], don) )
    print ('\n')
    i += 1


# In[108]:


# Party members donations 

cnad_amount.plot(kind='bar', figsize=(15,6))


# In[107]:


# SIngle candidate entry is less for Obama with respect otall republican party members

donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')


# In[110]:


# Donations and who they came from , A in occupations of the donors and create pivot
# and column to find for which party and find the asum of the contribution
# from the same occupation people

occupation = donor_df.pivot_table('contb_receipt_amt',
                                 index='contbr_occupation',
                                 columns='Party',
                                 aggfunc='sum')


# In[112]:


occupation


# In[113]:


# Number of Reported Occupation

occupation.shape


# In[114]:


# Cannot display all the contribution plot
# Contribution bigger than a million dollars

occupation = occupation[occupation.sum(1) > 1000000]

occupation.shape


# In[122]:


occupation.plot(kind='barh', figsize=(15,12), cmap='winter')


# In[124]:


#Combining same prfessions and drop wrong occupations

occupation.drop(['INFORMATION REQUESTED PER BEST EFFORTS',
                 'INFORMATION REQUESTED'], axis=0, inplace=True)


# In[125]:


occupation.shape


# In[127]:


occupation.loc['CEO'] = occupation.loc['CEO'] + occupation.loc['C.E.O.']


# In[129]:


occupation.drop('C.E.O.', inplace=True)


# In[130]:


occupation.plot(kind='barh',figsize=(15,12), cmap='winter' )

