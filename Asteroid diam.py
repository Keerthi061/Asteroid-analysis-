#!/usr/bin/env python
# coding: utf-8

# ##Importing Dataset

# In[82]:


import pandas as pd


# In[83]:


data = pd.read_csv('C:\\Users\\KEERTHI\\OneDrive\\Documents\\Preprocessing\\Asteroid_Updated.csv')    #importing dataset into object'data'


# ##Data provided

# In[84]:


noOfRows, noOfCol = data.shape
print("No of rows : "+ str(noOfRows))
print("No of columns "+ str(noOfCol))


# In[85]:


data.head() #Here we observe the data type inconsistency and values 


# In[86]:


data.tail() #lot of data is missing


# In[87]:


data.info() #Observe the non null count and the varrying datatype of values that the column contains


# We find that there is a lot of missing data. We also find that diameter needs to be converted to float format. Along with that, we also need to change categorical variables to integer format

# In[88]:


data.describe() #Getting an idea about the uniformity of the distribution and the difference between mimimum, maximum and mean values.


# The descriptions gives us a brief understanding about the data and it's distribution 

# In[89]:


data.hist(bins = 50, figsize = (25,25))


# We use graphs to better understand the data, but don't find the histogram for diamater, that's bec it's not in float datatype yet, thus, in the next step we do the conversion.

# In[90]:


converttofloat = {'diameter' : float}
data = data.astype(converttofloat) 


# In[91]:


data['diameter'].hist( figsize = (5,5))


# In[92]:


data['diameter'].describe()


# In[93]:


data['diameter'].median()


# The above two cells give us insights about the diameter. There's a large difference between the min and max value, and the mean or meadian don't do justice to all values of diameter. Thus, we will be dropping values instead of replacing them with the mean or median

# ##Understanding variables and Choosing a Target parameter

# In[94]:


'''
name - Asteroid's name
a -	semi-major axis[au] - half of the longest diameter of an ellipse.
e	- eccentricity - determines the amount by which its orbit around another body deviates from a perfect circle
i	- inclination wrt x-y ecliptic plane [deg]
om - longitude of the ascending node (angle from a specified reference direction, called the origin of longitude, to the direction of the ascending node)
w	- argument of perihelion -  angle from the body's ascending node to its periapsis
q	- perihelion (closest to sun) distance [au] - P=a(1−e) 
ad -	aphelion (farthest from sun) distance [au] - A=a(1+e) 
per_y	- orbital period [years]
data_arc	- data arc-span [d] (time span between earliest and latest observation)
condition_code	- orbit condition code (how well the orbit is known, 0- most well known, 9-least well known)
n_obs_used -	number of observations used
H - 	absolute magnitude parameter (apparent magnitude that the object would have if it were viewed from a distance of exactly 10 parsecs )
neo -	near earth object
pha -	physically hazardous asteroid 
diameter - 	diameter of asteroid [km]
extent -	object bi or tri-axial ellipsoid dimensions [km]
albedo - 	geometric albedo (the ratio of actual brightness as seen from the light source to that of an idealized flat, fully reflecting, diffusively scattering disk with the same cross-section.)
rot_per -	rotational period
gm -  (standard gravitational parameter) product of Gravitational constant and asteroid's mass (https://meetingorganizer.copernicus.org/EPSC-DPS2019/EPSC-DPS2019-1485-3.pdf) 
bv -	color index B-V magnitude difference -smaller, blue, hot
ub -	color index U-B magnitude difference 
IR -	color index I-R magnitude difference
spec_B - 	spectral taxonomic type (SMASSII)
spec_T -	spectral taxonomic type (Tholen)
G - Magnitude slope parameter
moid -	earth minimum orbit intersection distance [au]  (the distance between the closest points of the osculating orbits of two bodies)
class - classes of asteroid
n-  rotation axis orientation (https://issfd.org/ISSFD_1999/pdf/ODY_4.pdf) 
'''


# ##Target Parameter - Diameter
# 
# Our main task is to predict and caliber the magnitude of asteroids. To do that, we have to choose a target parameter, for the estimation of which, we'll test various ML Models. After going through the data provided and understanding the task and the significance of each variable, it is understood that the diameter poses as one of the most significant parameters. The magnitude of the diameter demonstrates the magnitude of the asteroids. It is also directly related to many other parameters like the semi-major axis (a), absolute magnitude (H),  albedo, and more. Moreover, it can also be used to tell if an asteroid is a potentially hazardous one or not.
# 
# Thus, the target parameter chosen - Diameter

# ##Checking correlation between data variables

# In[95]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[96]:


corrmat =  data.corr()
f, ax =  plt.subplots(figsize=(10,10))
sns.heatmap(corrmat,vmax=0.8,square=True, annot=True, fmt='.2f');
plt.show()


# The correlation matrix don't give us many good insights as the data is unclean. But we find some uncanny behaviour in the IR row and column and also see that some columns have high multicolinearity. We'll be dealing with that after cleaning our data
# 
# 
# 
# 
# 
# 

# ##Preparing the dataset
# 
# This includes cleaning and transforming raw data into useful information for further analysis and processing
# 
# In this step we clean, prepare the dataset by handling missing values and substituting particular values whenever needed.
# 

# ####HANDLING MISSING VALUES

# In[97]:


missing_values_count = data.isnull().sum()
print(missing_values_count.sort_values())


# A lot of our data is missing. Let's find what percentage is, to get a better understanding.

# In[98]:


total_cells = np.product(data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(str(percent_missing)+" percent of our data is missing.")


# So, let's check ten unique values from each column

# In[99]:


for column in data.columns:
    print(column, data[column].unique()[:10])


# In[100]:


data.corr()['diameter'].abs().sort_values(ascending=False)


# The above cell's output show the corelation of other values with the values of diameter. Thus, we must make sure to include diameter, GM, H, data_arc, n_obs_used, moid, q, BV and n, UB and a in our model

# Before that, let's drop some missing values

# In[101]:


data1 = data


# In[102]:


data1.dropna()


# We can't drop all missing values because then we'll lose all our data.

# So, let's try dropping columns with more than 1 missing value

# In[103]:


data


# In[104]:


data1 = data


# In[105]:


data=data1


# In[106]:


print(missing_values_count.sort_values())


# In[107]:


columns_without_na = data1.dropna(axis=1)
columns_without_na.head()


# In[108]:


print("Columns in original dataset: %d \n" % data.shape[1])
print("Columns with na's dropped: %d" % columns_without_na.shape[1])


# But this way we lose most of our important data. So, let's manually remove columns that we can drop.

# In[109]:


data.drop(['ad','i','e','per','per_y','G','ma','rot_per','w','om','IR'], axis=1, inplace=True)


# We could remove na values now but but we have only 14 non null GM values, 1021 BV values and 979 UB values. So, let's understand these data.

# In[110]:


data['GM'].describe()


# In[111]:


data['BV'].describe()


# In[112]:


data['UB'].describe()


# The data looks quite uniform, thus we'll fill missing values of these columns with the mean.

# In[113]:


data['GM'] = data['GM'].fillna((data['GM'].mean()))


# In[114]:


data['BV'] = data['BV'].fillna((data['BV'].mean()))


# In[115]:


data['UB'] = data['UB'].fillna((data['UB'].mean()))


# In[116]:


data


# Now, let's drop columns of non-numeric type

# In[117]:


data.drop(['name','condition_code','neo','extent','spec_B','spec_T','class'], axis=1, inplace=True)


# We dropped features like name, condition_code, neo and more because the asteroid's name or, where the asteroid lies or, whether it is near earth or not or, the taxonomy has nothing to do with the diameter.<br>
# 

# In[118]:


data


# In[119]:


data = data.dropna(subset=['BV','diameter','H', 'albedo','data_arc']) #dropping records with these as Nan


# In[120]:


data.isna().values.any() #Checking if there are null values


# In[121]:


data


# In[122]:


print(data.isnull().sum()) #Checking how many null values are present


# Since pha is the only column with categorical data, we need to convert it to int and perform categorical encoding.

# In[123]:


pha_entries = data['pha'].unique()  #finding out unique data to encode
pha_entries.sort()
print(pha_entries)


# categorical encoding

# In[124]:


cleanup_pha = {"pha": {"N": 0, "Y": 1, }}
data = data.replace(cleanup_pha) #replacing N by 0 and Y by 1
data.head()


# In[125]:


data.info() #checking if any more conversion is required


# In[126]:


#checking the correlation heatmap to gain some insight on the now clean data
corrmat =  data.corr()
f, ax =  plt.subplots(figsize=(10,10))
sns.heatmap(corrmat,vmax=0.8,square=True, annot=True, fmt='.2f');
plt.show()


# Analzying from the heatmap matrix, we deicde to drop n_obs_used and data_arc

# In[127]:


data.drop(['n_obs_used'], axis=1, inplace=True)


# In[128]:


data


# In[129]:


corrmat =  data.corr()
f, ax =  plt.subplots(figsize=(10,10))
sns.heatmap(corrmat,vmax=0.8,square=True, annot=True, fmt='.2f');
plt.show()


# In[130]:


cleandata = data


# ##Splitting data into model features and the target

# In[131]:


y = cleandata['diameter'] #target
X = cleandata.drop(['diameter'],axis = 1) #features


# In[132]:


X = X.iloc[:,:].values
X.shape #checking the shape of the feature data


# ##Feature Scaling

# In[133]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[134]:


transf = StandardScaler()
data = transf.fit_transform(data)
X_std = transf.fit_transform(X)


# In[135]:


from pandas import DataFrame


# In[136]:


dataset = DataFrame(data)


# In[137]:


dataset.hist()


# ##Train- Test splitting of the data
# 
# 

# In[138]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_std, y, test_size = 0.2, random_state = 0)


# ##Tring different ML algorithms

# In[139]:


from sklearn.metrics import mean_squared_error      #for getting the mean squared error
from sklearn.metrics import r2_score         #to get the accuracy of each model


# In[140]:


import seaborn as sns

def plot(prediction):   #For plotting prediction and visualizing how well the model is doing
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(9,5)) 
    sns.distplot(Y_test.values,label='Test Values', ax=ax1)
    sns.distplot(prediction ,label='Prediction', ax=ax1)
    ax1.set_xlabel('Distribution Plot')
    ax2.scatter(Y_test,prediction, c='yellow',label='Predictions')
    ax2.plot(Y_test,Y_test,c='blue',label='y=x')
    ax2.set_xlabel('test value')
    ax2.set_ylabel('estimated $\log(radius)$')
    ax1.legend()
    ax2.legend()
    ax2.axis('scaled') #same x y scale

algorithmsList=[]  #list that will contain all algorithm names
rmseList=[]        #list with the root mean squared error for each algorithm
modelScore=[]  #list with the R2 score of each model


# ####LINEAR REGRESSION

# In[141]:


algorithmsList.append("Linear Regression")

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ####DECISION TREE

# In[142]:


algorithmsList.append("Decision Tree")

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ####RANDOM FOREST
# 
# 
# 
# 

# In[143]:


algorithmsList.append("Random Forest")

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ####ELASTIC NET CV

# In[144]:


algorithmsList.append("Elastic Net CV")

from sklearn.linear_model import ElasticNetCV
model = ElasticNetCV()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###KNN

# In[145]:


algorithmsList.append("KNN")

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                  
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###RIDGE

# In[146]:


algorithmsList.append("Ridge")

from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###MLP Regression

# In[147]:


algorithmsList.append("MLP Regressor")

from sklearn.neural_network import MLPRegressor
model = MLPRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###Lasso

# In[148]:


algorithmsList.append("Lasso")

from sklearn import linear_model
model = linear_model.Lasso(alpha=0.1)
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###LGBM Regression

# In[149]:


algorithmsList.append("LGBM")

from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###XGBoost

# In[150]:


algorithmsList.append("XG Boost")

from xgboost.sklearn import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###CatBoost Regression

# In[151]:


get_ipython().system('pip3 install catboost')


# In[152]:


algorithmsList.append("CatBoost")

from catboost import CatBoostRegressor
model = CatBoostRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###Bayesian Ridge

# In[153]:


algorithmsList.append("Bayesian Ridge")

from sklearn.linear_model import BayesianRidge
model = BayesianRidge()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###Gradient Boosting 

# In[154]:


algorithmsList.append("Gradient Boosting")

from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ###Support Vector Machine

# In[155]:


algorithmsList.append("Support Vector Machine")

from sklearn.svm import SVR
model = SVR()
model.fit(X_train, Y_train)
model.predict(X_test)

diameterPrediction  = model.predict(X_test)
mse = mean_squared_error(Y_test, diameterPrediction)
rmse = np.sqrt(mse)
print("root mean square error : "+str(rmse))

rmseList.append(rmse)

r2 = r2_score(Y_test,diameterPrediction)
print("R2 Score : ",r2)                     #This shows the score of how well the model has fit 
print("\n\n\n")

modelScore.append(r2)

plot(diameterPrediction)


# ##CONCLUSION

# In[156]:


import matplotlib.pyplot as plt


# In[157]:


len(modelScore)


# In[158]:


len(algorithmsList)


# In[159]:


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(algorithmsList, modelScore, color ='blue', width = 0.4)
plt.xticks(rotation = 45)
plt.xlabel("algorithms tested")
plt.ylabel("model score")
plt.title("ml")
plt.show()


# ###Conclusion
# **Decision Tree Regression** with the score of 0.9790032781 shows the best performance.
