
# coding: utf-8
'''
First some basic questions:

1.) Who were the passengers on the Titanic? (Ages,Gender,Class,..etc)
2.) What deck were the passengers on and how does that relate to their class?
3.) Where did the passengers come from?
4.) Who was alone and who was with family?

Then we'll dig deeper, with a broader question:

5.) What factors helped someone survive the sinking?

'''
# In[1]:


import pandas as pd
from pandas import Series , DataFrame


# In[2]:


titanic_df = pd.read_csv('train.csv')


# In[3]:


titanic_df.head()


# In[4]:


titanic_df.info()


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


#ANSWERING QUESTION 1

sns.factorplot(x='Sex', data=titanic_df, kind='count')


# In[7]:


#hue defines each subtype in the given column with different color
#To distinguish a certain group using differnet color

sns.factorplot('Sex', data=titanic_df, kind='count', hue='Pclass')


# In[8]:


sns.factorplot('Pclass', data=titanic_df, kind='count', hue='Sex')


# In[9]:


#Assuming child age will be below 16

def mfc(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex


# In[10]:


#apply a function to dataset by creating a new column named=person

titanic_df['person'] = titanic_df[['Age','Sex']].apply(mfc, axis=1)
titanic_df.head()


# In[11]:


titanic_df[0:10]


# In[12]:


sns.factorplot('Pclass' , data=titanic_df, kind='count', hue='person')


# In[13]:


titanic_df['Age'].hist(bins=70)


# In[14]:


titanic_df['Age'].mean()


# In[15]:


titanic_df['person'].value_counts()


# In[16]:


#Multiple types of graphs in same Grid

fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)
fig.map(sns.kdeplot, 'Age', shade= True)

oldest = titanic_df['Age'].max()

fig.set(xlim= (0, oldest))

fig.add_legend()


# In[17]:


#Multiple types of graphs in same Grid

fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot, 'Age', shade= True)

oldest = titanic_df['Age'].max()

fig.set(xlim= (0, oldest))

fig.add_legend()


# In[18]:


#Multiple types of graphs in same Grid

fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot, 'Age', shade= True)

oldest = titanic_df['Age'].max()
fig.set(xlim= (0, oldest))

fig.add_legend()


# In[19]:


#ANSWERING QUESTION 2 


# In[20]:


titanic_df.head()


# In[21]:


#removing null vlaues, creating new column

deck = titanic_df['Cabin'].dropna()


# In[22]:


deck.head()


# In[23]:


#Grabbing deck character (First letter) , defining different cabins.

levels = []

for level in deck:
    levels.append(level[0])
    
cabin_df = DataFrame(levels)

#palette defining the colour of the plot shaded in winter format
#The below link defines different color shades in 'matplotlib'
#https://matplotlib.org/examples/color/colormaps_reference.html

cabin_df.columns = ['Cabin']
sns.factorplot('Cabin', data=cabin_df.sort_values('Cabin'), palette='winter_d', 
               kind='count')


# In[24]:


cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.factorplot('Cabin', data=cabin_df.sort_values('Cabin'), palette='summer', 
               kind='count')


# In[25]:


titanic_df.head()


# In[26]:


#ANSWERING QUESTION 3


# In[27]:


# Three places from where the people came form : 
# C = Cherbourg, Q = Queenstown, S = Southampton

sns.factorplot('Embarked', data=titanic_df, hue='Pclass', 
               kind='count',order=['C', 'Q', 'S'])


# In[28]:


#ANSWERING QUESTION 4


# In[29]:


#As Fmaily 2 Categories are SibSp and Parch
# SibSp = Sibling Special and Parch = Parent Child
# If both Columns are empty that means that the passenger came alone
# Else with a family


# In[30]:


titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[31]:


# Anything other than 0 are with family

titanic_df['Alone']


# In[32]:


#Warning that changing the original dataset

titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'


# In[33]:


titanic_df.head()


# In[34]:


#Most people are alone than with familt

sns.factorplot('Alone', data=titanic_df, palette='Blues',
               kind='count')


# In[35]:


#ANSWERING QUESTION 5


# In[36]:


#Survived Column chaning to alphanumeric values

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no' , 1:'yes'})

sns.factorplot('Survivor', data=titanic_df, palette='Set2', kind='count')


# In[37]:


#What Factors affecting the Survivor Rate:

sns.factorplot('Pclass', 'Survived', data=titanic_df)


# In[41]:


sns.factorplot('Pclass', 'Survived', data=titanic_df, hue='person')


# In[42]:


#Linear Regression plot for the survivors

sns.lmplot('Age', 'Survived', data=titanic_df)


# In[43]:


sns.lmplot('Age', 'Survived', data=titanic_df, 
           hue='Pclass', palette='winter')


# In[44]:


generations = [10,20,40,60,80]

sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df
           ,palette='winter', x_bins=generations )


# In[45]:


sns.lmplot('Age', 'Survived', hue='Sex', data=titanic_df
          ,palette='winter', x_bins=generations)

