#!/usr/bin/env python
# coding: utf-8

# In[124]:


import pandas as pd # import pandas through pd
df = pd.DataFrame() # create a dataframe by calling the DataFrame method from pandas
# call the read_csv method from python to read the csv file and save it into DataFrame
df = pd.read_csv('/Users/aadityachangayil/Downloads/adult.csv') 

# groupby python function will group the parts of the data, like how the below dataset groups
# the ages by class. Then find the mean of the data set. 
df1 = df.groupby(["sclass"])[["age"]].mean() 
print(df1)

# the function here is grouping both hours per weeek and age by class. Then find the standard 
# deviation of the data set.
df2 = df.groupby(["sclass"])[["hours-per-week","age"]].std()
print(df2)

# the same as the last one, except find the correlation of the set
df3 = df.groupby(["sclass"])[["hours-per-week","age"]].corr()
print(df3)


     


# In[119]:


import pandas as pd
df = pd.DataFrame()
df = pd.read_csv('/Users/aadityachangayil/Downloads/car.csv')

# pandas method replace goes to methods based on the title for the columns, and 
# in that, finds each unique name, and replaces it with a number. inplace = True is 
# passed as a parameter to make the change permanent. If not added, no change will happen.
df.replace({'Buying': {'low': 1, 'med': 3, 'high': 5, 'vhigh': 7}}, inplace = True)
df.replace({'Maintenance': {'low': 1, 'med': 3, 'high': 5, 'vhigh': 7}},inplace = True)
df.replace({'Doors': {'5more': 5}}, inplace = True)
df.replace({'Riders': {'more': 6}}, inplace = True)
df.replace({'Trunk_Size': {'small': 2, 'med': 4, 'big': 6}}, inplace = True)
df.replace({'Safety': {'low': 1, 'med': 3, 'high': 5}}, inplace = True)
df.replace({'Class': {'unacc': 0, 'acc': 2, 'good': 4, 'vgood': 6}}, inplace = True)
print(df) 

dfc = df[['Buying','Safety','Maintenance']].corr() # print the correlation for all 3 items.
dfc # in python, you can type the variable at the end and it will print itself without you 
# having to type print(). It only happens if you want to print something at the end.





# In[123]:


import pandas as pd
df = pd.DataFrame()
df = pd.read_csv('/Users/aadityachangayil/Downloads/iris.csv')

df2 = df.groupby(["Flower_class"])[["Petal_length","Petal_width"]].median()
print(df2)

# if you call the correlation method for Flower_class, it will print the correlation between 
# all methods in the class without you having to type anything.
dfc = df.groupby(["Flower_class"]).corr() 
dfc


# In[ ]:




