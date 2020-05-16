
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
import time
import datetime
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,mean_absolute_error
import datetime
import operator
plt.style.use('seaborn')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


confirmed_cases= pd.read_csv('C:/Users/divya.a/Desktop/AIProj/time_series_covid_19_confirmed.csv')
confirmed_cases.head()


# In[3]:


deaths_cases= pd.read_csv('C:/Users/divya.a/Desktop/AIProj/time_series_covid_19_deaths.csv')
deaths_cases.head()


# In[4]:


recovered_cases= pd.read_csv('C:/Users/divya.a/Desktop/AIProj/time_series_covid_19_recovered.csv')
recovered_cases.head()


# In[5]:


cols = confirmed_cases.keys()
cols


# In[6]:


confirmed = confirmed_cases.loc[:, cols[4]:cols[-1]]


# In[7]:


deaths = deaths_cases.loc[:, cols[4]:cols[-1]]


# In[8]:


recoveries = recovered_cases.loc[:, cols[4]:cols[-1]]


# In[9]:


confirmed.head()


# In[10]:


dates = confirmed.keys()
world_cases = []
total_deaths = []
motality_rate = []
total_recovered = []

for i in dates:
  confirmed_sum = confirmed[i].sum()
  death_sum = deaths[i].sum()
  recovered_sum = recoveries[i].sum()
  world_cases.append(confirmed_sum)
  total_deaths.append(death_sum)
  motality_rate.append(death_sum/confirmed_sum)
  total_recovered.append(recovered_sum)


# In[11]:


confirmed_sum


# In[12]:


death_sum 


# In[13]:


recovered_sum 


# In[14]:


world_cases


# In[15]:


days_since_1_22=np.array([i for i in range(len(dates))]).reshape(-1,1)
world_cases = np.array(world_cases).reshape(-1,1)
total_deaths=np.array(total_deaths).reshape(-1,1)
total_recovered=np.array(total_recovered).reshape(-1,1)
days_since_1_22


# In[16]:


world_cases


# In[17]:


total_deaths


# In[18]:


days_in_future = 10
future_forecast = np.array([i for i in range(len(dates)+days_in_future)]).reshape(-1,1)
adjusted_dates = future_forecast[:-10]
future_forecast


# In[19]:


import datetime
start = '1/22/2020'
start_date = datetime.datetime.strptime(start, '%m/%d/%Y')
future_forecast_dates = []
for i in range(len(future_forecast)):
    future_forecast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))


# In[20]:


latest_confirmed = confirmed_cases[dates[-1]]
latest_deaths = deaths_cases[dates[-1]]
latest_recovered = recovered_cases[dates[-1]]
latest_confirmed


# In[21]:


latest_deaths


# In[22]:


latest_recovered


# In[23]:


unique_countries = list(confirmed_cases['Country/Region'].unique())
unique_countries


# In[24]:


country_confirmed_cases = []
no_cases = []
for i in unique_countries:
    cases = latest_confirmed[confirmed_cases['Country/Region']==i].sum()
    if cases>0:
        country_confirmed_cases.append(cases)
    else:
        no_cases.append(i)

for i in no_cases:
    unique_countries.remove(i)

unique_countries = [k for k, v in sorted(zip(unique_countries, country_confirmed_cases), key=operator.itemgetter(1),reverse=True)]
for i in range(len(unique_countries)):
    country_confirmed_cases[i] = latest_confirmed[confirmed_cases['Country/Region']==unique_countries[i]].sum()


# In[25]:


print('confimed cases : Country/Regions')
for i in range(len(unique_countries)):
    print(f' {unique_countries[i]} : {country_confirmed_cases[i]} cases')


# In[26]:


uni_pro=list(confirmed_cases['Province/State'].unique())
print(uni_pro)


# In[27]:


province_confirmed_cases=[]
no_cases=[]
for i in uni_pro:
    case=latest_confirmed[confirmed_cases['Province/State']==i].sum()
    if case>0:
        province_confirmed_cases.append(case)
    else:
        no_cases.append(i)
        
for i in no_cases:
    uni_pro.remove(i)


# In[28]:


for i in range(len(uni_pro)):
    print(f'{uni_pro[i]}: {province_confirmed_cases[i]} cases')


# In[29]:


nan_indices=[]

for i in range(len(uni_pro)):
    if type(uni_pro) == float:
        nan_indices.append(i)
        
uni_pro=list(uni_pro)
province_confirmed_cases=list(province_confirmed_cases)

for i in nan_indices:
    uni_pro.pop(i)
    province_confirmed_cases.pop(i)


# In[31]:


plt.figure(figsize=(34,34))
plt.barh(unique_countries,country_confirmed_cases)
plt.title("Number of covid19 confirmed cases in countries")
plt.xlabel("number of covid confirm cases")
plt.show()


# In[32]:


china_conf=latest_confirmed[confirmed_cases['Country/Region'] =='China'].sum()
out_of_china_conf=np.sum(country_confirmed_cases)- china_conf
plt.figure(figsize=(16,9))
plt.barh('China',china_conf)
plt.barh('out_of_china',out_of_china_conf)
plt.title('no. of confirmed cases')
plt.show()


# In[33]:


print('out of china {} cases:'.format(out_of_china_conf))
print('in china: {} cases'.format(china_conf))
print('Total : {} cases '.format(out_of_china_conf + china_conf))


# In[38]:


visual_uni_cont = []
visual_confirmed_cases=[]
others = np .sum(country_confirmed_cases[10:])
for i in range(len(country_confirmed_cases[:10])):
    visual_uni_cont.append(unique_countries[i])
    visual_confirmed_cases.append(country_confirmed_cases[i])
    
visual_uni_cont.append('others')    
visual_confirmed_cases.append(others)


# In[39]:


plt.figure(figsize=(32,20))
plt.barh(visual_uni_cont,visual_confirmed_cases)
plt.title('Number of covid19 confirmed cases in country/region',size=20)
plt.show()


# In[41]:


c=random.choices(list(mcolors.CSS4_COLORS.values() ), k=len(unique_countries))
plt.figure(figsize=(10,10))
plt.title("covid19 confirmed cases")
plt.pie(visual_confirmed_cases,colors=c)
plt.legend(visual_uni_cont,loc="best")
plt.show()


# In[44]:


X_train_confirmed,X_test_confirmed,y_train_confirmed,y_test_confirmed=train_test_split(days_since_1_22,world_cases,test_size=0.15,shuffle=False)


# In[66]:


#kernel=['poly','signoid','rbf']
#c=[0.01,0.1,1,10]
#gamma=[0.01,0.1,1]
#epsilon=[0.01,0.1,1]
#shrinking=[True,False]
#svm_grid={'kernel':kernel ,'c': c ,'gamma' : gamma , 'shrinking' : shrinking}

#svm=SVR()
#svm_search=RandomizedSearchCV(svm, svm_grid, scoring='neg_mean_squared_error' ,cv=3 ,return_train_score=True,n_jobs=-1,n_iter=40,verbose=1)
#svm_search.fit(X_train_confirmed,y_train_confirmed)


# In[67]:


#svm_search.best_params


# In[50]:


#linear regression
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(normalize=True ,fit_intercept=True)


linear_model.fit(X_train_confirmed,y_train_confirmed)
test_linear_pred= linear_model.predict(X_test_confirmed)
linear_pred =linear_model.predict(future_forecast)
print('MAE : ',mean_absolute_error(test_linear_pred,y_test_confirmed))
print('MSE : ',mean_squared_error(test_linear_pred,y_test_confirmed))


# In[50]:


#linear regression
from sklearn.linear_model import LinearRegression
linear_model=LinearRegression(normalize=True ,fit_intercept=True)


linear_model.fit(X_train_confirmed,y_train_confirmed)
test_linear_pred= linear_model.predict(X_test_confirmed)
linear_pred =linear_model.predict(future_forecast)
print('MAE : ',mean_absolute_error(test_linear_pred,y_test_confirmed))
print('MSE : ',mean_squared_error(test_linear_pred,y_test_confirmed))


# In[51]:


plt.plot(y_test_confirmed)
plt.plot(test_linear_pred)


# In[54]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,world_cases)
plt.plot(future_forecast,linear_pred,linestyle='dashed',color='green')
plt.title('number of corona cases over time ',size=27)
plt.xlabel('days since 22-mar-2020' ,size=27)
plt.ylabel('no of cases',size=27)
plt.legend(['confirmed cases' ,'linear regression predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[55]:


print('linear regression future predictions')
print(linear_pred[-15:])


# In[57]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_deaths,color='red')
plt.title('number of corona cases over time ',size=27)
plt.xlabel('time' ,size=27)
plt.ylabel('no of deaths',size=27)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[60]:


mean_motality_rate=np.mean(motality_rate)
plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,motality_rate,color='pink')
plt.axhline(y=mean_motality_rate,linestyle='dashed',color='green')
plt.title('mortality rate of covid19 over time ',size=27)
plt.xlabel('days since 22-mar-2020' ,size=27)
plt.ylabel('Mortality rate',size=27)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[61]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_recovered, color='blue')
plt.title('number of corona cases  recoveredover time ',size=27)
plt.xlabel('days since 22-mar-2020' ,size=27)
plt.ylabel('no of cases',size=27)
plt.legend(['confirmed cases' ,'linear regression predictions'])
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[62]:


plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_deaths,color='red')
plt.plot(adjusted_dates,total_recovered,color='green')
plt.title('number of corona cases over time ',size=27)
plt.xlabel('days since 22-mar-2020' ,size=27)
plt.ylabel('no of cases',size=27)
plt.legend(['deaths' ,'recoveries'],loc='best',fontsize=20)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()


# In[65]:


#deaths vs recoveries

plt.figure(figsize=(20,12))
plt.plot(adjusted_dates,total_recovered,total_deaths)

plt.title('covid19 deaths vs recoveries ',size=27)
plt.xlabel('Total Deaths' ,size=27)
plt.ylabel('Total Recoveries',size=27)
plt.xticks(size=15)
plt.yticks(size=15)
plt.show()

