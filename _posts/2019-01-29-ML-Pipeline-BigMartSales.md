---
title: "First ML Pipeline: BigMart Sales"
date: 2019-01-29
tags: [machine learning]
header:
  image: "/images/MLPipeline-BigMartSales/sale.jpg"
excerpt: "Data cleaning, Machine learning, Data Science"
mathjax: "true"
---

# ML techniques introduction

Aim: predict sales of certain products in certain stores for a supermarket chain - 
while doing so, we will also learn about the various tools and techniques used in a 
data science pipeline. 

We will be following the tutorial given [here](https://www.analyticsvidhya.com/blog/2016/02/bigmart-sales-solution-top-20/)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## 1. Data Exploration

```python
trainData = pd.read_csv('./Data/Train_UWu5bXk.csv')
testData = pd.read_csv('./Data/Test_u94Q5KV.csv')

# initially we can combine the data to carry out 
# feature engineering and then can split it for training later

# introduce a new column, 'source' to keep track of the data
trainData['source'] = 'train'
testData['source'] = 'test'

# combine data sets
allData = pd.concat([trainData, testData], ignore_index=True)

# let's check the columns in allData for missing values
print(allData.apply(lambda x: sum(x.isnull())))
print('size of all data', allData.shape)
print('size of all train data', trainData.shape)
print('size of all test data', testData.shape)
```

    Item_Fat_Content                0
    Item_Identifier                 0
    Item_MRP                        0
    Item_Outlet_Sales            5681
    Item_Type                       0
    Item_Visibility                 0
    Item_Weight                  2439
    Outlet_Establishment_Year       0
    Outlet_Identifier               0
    Outlet_Location_Type            0
    Outlet_Size                  4016
    Outlet_Type                     0
    source                          0
    dtype: int64
    size of all data (14204, 13)
    size of all train data (8523, 13)
    size of all test data (5681, 12)


We don't need to be worried about Item_outlet_sales - 
the empty values belong to the test data, but item weight 
and outlet_size is something to think about

```python
# lets see the tyes of data in each column
allData.dtypes 
```
    Item_Fat_Content              object
    Item_Identifier               object
    Item_MRP                     float64
    Item_Outlet_Sales            float64
    Item_Type                     object
    Item_Visibility              float64
    Item_Weight                  float64
    Outlet_Establishment_Year      int64
    Outlet_Identifier             object
    Outlet_Location_Type          object
    Outlet_Size                   object
    Outlet_Type                   object
    source                        object
    dtype: object

```python
# general stats on the data (float only)
allData.describe()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Item_MRP</th>
      <th>Item_Outlet_Sales</th>
      <th>Item_Visibility</th>
      <th>Item_Weight</th>
      <th>Outlet_Establishment_Year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14204.000000</td>
      <td>8523.000000</td>
      <td>14204.000000</td>
      <td>11765.000000</td>
      <td>14204.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>141.004977</td>
      <td>2181.288914</td>
      <td>0.065953</td>
      <td>12.792854</td>
      <td>1997.830681</td>
    </tr>
    <tr>
      <th>std</th>
      <td>62.086938</td>
      <td>1706.499616</td>
      <td>0.051459</td>
      <td>4.652502</td>
      <td>8.371664</td>
    </tr>
    <tr>
      <th>min</th>
      <td>31.290000</td>
      <td>33.290000</td>
      <td>0.000000</td>
      <td>4.555000</td>
      <td>1985.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>94.012000</td>
      <td>834.247400</td>
      <td>0.027036</td>
      <td>8.710000</td>
      <td>1987.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>142.247000</td>
      <td>1794.331000</td>
      <td>0.054021</td>
      <td>12.600000</td>
      <td>1999.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>185.855600</td>
      <td>3101.296400</td>
      <td>0.094037</td>
      <td>16.750000</td>
      <td>2004.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>266.888400</td>
      <td>13086.964800</td>
      <td>0.328391</td>
      <td>21.350000</td>
      <td>2009.000000</td>
    </tr>
  </tbody>
</table>
</div>

Good idea would be to change outlet_est_year to age of company
Additionally, the minimum item_vis of value 0.0 does not make sense
The different sd's for the features needs to be noted incase we use
non-scale invariant ML techniques

```python
# info about features that are categorical rather than float
# can look at the # of unique values in each feature

allData.apply(lambda x: len(x.unique()))
```

    Item_Fat_Content                 5
    Item_Identifier               1559
    Item_MRP                      8052
    Item_Outlet_Sales             3494
    Item_Type                       16
    Item_Visibility              13006
    Item_Weight                    416
    Outlet_Establishment_Year        9
    Outlet_Identifier               10
    Outlet_Location_Type             3
    Outlet_Size                      4
    Outlet_Type                      4
    source                           2
    dtype: int64

let's consider specifically Item_Fat_Content, Item_Type & outlet_identifier, location type, size and type

```python
# need to create a list of column names which are of type object 
# (string)
object_columns = [x 
                  for x in allData.dtypes.index 
                  if allData.dtypes[x]=='object']

# remove item_identifier. outlet identifier and source
for i in ['Item_Identifier', 'Outlet_Identifier', 'source']:
    object_columns.remove(i)

# print the freq of each unique item in the object_columns list
for col in object_columns:
    print(col, '\n', allData[col].value_counts(), '\n')
```

    Item_Fat_Content 
     Low Fat    8485
    Regular    4824
    LF          522
    reg         195
    low fat     178
    Name: Item_Fat_Content, dtype: int64 
    
    Item_Type 
     Fruits and Vegetables    2013
    Snack Foods              1989
    Household                1548
    Frozen Foods             1426
    Dairy                    1136
    Baking Goods             1086
    Canned                   1084
    Health and Hygiene        858
    Meat                      736
    Soft Drinks               726
    Breads                    416
    Hard Drinks               362
    Others                    280
    Starchy Foods             269
    Breakfast                 186
    Seafood                    89
    Name: Item_Type, dtype: int64 
    
    Outlet_Location_Type 
     Tier 3    5583
    Tier 2    4641
    Tier 1    3980
    Name: Outlet_Location_Type, dtype: int64 
    
    Outlet_Size 
     Medium    4655
    Small     3980
    High      1553
    Name: Outlet_Size, dtype: int64 
    
    Outlet_Type 
     Supermarket Type1    9294
    Grocery Store        1805
    Supermarket Type3    1559
    Supermarket Type2    1546
    Name: Outlet_Type, dtype: int64 

For fat content - Low Fat and LF and low fat all belong to the same class
same can be said for reg and Regular. For item_type - maybe combining certain 
types of items may be better - like all drinks

## 2. Data Cleaning

We will use the data we have to fill in the missing values for item weight and outlet size

For missing values that are continuous the best thing to do is to take a mean of similar 
items and replace the missing values for the corresponding mean

For categorical missing values it is best to consider mode.

Using both a boolean array (locating the missing values) and a pd series which contains 
the means or modes, we can then replace the missing values in the pd df using loc - see below.

```python
# for the 2439 missing item weight values we can can average over the 
# specific item identifiers and fill in the missing values accordingly

# first we need the average weight for each item identifier:
uniqueItems = allData['Item_Identifier'].unique()

averageWeights = [allData.loc[
                      allData['Item_Identifier'] == j, 
                                'Item_Weight'
                             ].mean() 
                  for j in uniqueItems]

# create pandas series from average weights with index label being unique items
averageWeightsDS = pd.Series(averageWeights, index=uniqueItems)

# create a boolean array which locates the missing item weights
missingWeigths = allData['Item_Weight'].isnull()

# in the allData df, replace the missing elements with averages corresponding
# to the item_identifier
allData.loc[missingWeigths, 'Item_Weight'] = \
    allData.loc[missingWeigths, 'Item_Identifier'].apply(
        lambda x: averageWeightsDS[x])    
```

```python
# for the 4016 missing values for outlet size - we can use the mode 
# of the outlet_types

from scipy.stats import mode

uniqueOutletTypes = allData['Outlet_Type'].unique()

#for each of the outlet types we need to determine the mode:
modeOutletSize = []
for outletType in uniqueOutletTypes:
    outletTypeList = allData.loc[
                    allData['Outlet_Type']==outletType,'Outlet_Size']
    mode = outletTypeList.mode()[0]
    modeOutletSize.append(mode)
    
# create a series with mode info
modeOutletSize = pd.Series(modeOutletSize, index = uniqueOutletTypes)
    
# crendate a boolean indicating the positions of the missing outlet size
# values
missingOSValues = allData['Outlet_Size'].isnull()

# use the boolean and the modeOutletSize to fill in the missing values
allData.loc[missingOSValues, 'Outlet_Size'] = \
    allData.loc[missingOSValues, 'Outlet_Type'].apply(
    lambda x: modeOutletSize[x])
```

There are some items that have 0.0 visibility - this does not make sense
if they are being sold in a store - therefore we can take the unique
Item_Identifier and create an average visibility for each Item_Identifier - 
this can then be used along with a boolean to remove 0.0 visibilty values

```python
# first we need the average weight for each item identifier:
uniqueItems = allData['Item_Identifier'].unique()

averageVis = [allData.loc[
                      allData['Item_Identifier'] == j, 
                                'Item_Visibility'
                             ].mean() 
                  for j in uniqueItems]

averageVis = pd.Series(averageVis, index = uniqueItems)

# create boolean locating 0.0 item visibilty
zeroVisBoolean = (allData['Item_Visibility']==0)

#replace the 0.0 elements with averageVis
allData.loc[zeroVisBoolean, 'Item_Visibility'] = \
allData.loc[zeroVisBoolean, 'Item_Identifier']. apply(
lambda x: averageVis[x])
```

## 3. Feature Engineering

There are 16 Item types:

```python
allData['Item_Type'].value_counts()
```

    Fruits and Vegetables    2013
    Snack Foods              1989
    Household                1548
    Frozen Foods             1426
    Dairy                    1136
    Baking Goods             1086
    Canned                   1084
    Health and Hygiene        858
    Meat                      736
    Soft Drinks               726
    Breads                    416
    Hard Drinks               362
    Others                    280
    Starchy Foods             269
    Breakfast                 186
    Seafood                    89
    Name: Item_Type, dtype: int64

We could possibly combine certain items together 
In the Item_Identifier which is a unique item id for each item, 
we can see id's beginning with FD, DR or NC - which probably stand
for Food, Drinks and Non-consumables - so we can make a custom
item type column

```python
# new column to be based on the first two elements of the item
# identifier string
allData['New_Item_Type'] = allData['Item_Identifier'].apply(
                            lambda x: x[0:2])

allData['New_Item_Type'].value_counts()
```
    FD    10201
    NC     2686
    DR     1317
    Name: New_Item_Type, dtype: int64

As mentioned above, we should also have a column that describes 
the number of years a store has been open rather then the outlet 
establishment year (year data was recorded is 2013)

```python
allData['Outlet_Age'] = allData['Outlet_Establishment_Year'].apply(
                        lambda x: 2013-x)
allData['Outlet_Age'].describe()
```

    count    14204.000000
    mean        15.169319
    std          8.371664
    min          4.000000
    25%          9.000000
    50%         14.000000
    75%         26.000000
    max         28.000000
    Name: Outlet_Age, dtype: float64

As mentioned previously, the fat content labels have some overlap so it 
would be better to combine some values

```python
allData['Item_Fat_Content'].value_counts()
```

    Low Fat    8485
    Regular    4824
    LF          522
    reg         195
    low fat     178
    Name: Item_Fat_Content, dtype: int64

```python
allData['Item_Fat_Content'] = allData['Item_Fat_Content'].replace(
            {'LF':'Low Fat', 'low fat':'Low Fat', 
             'reg':'Regular'})
allData['Item_Fat_Content'].value_counts()
```

    Low Fat    9185
    Regular    5019
    Name: Item_Fat_Content, dtype: int64

But all items should not be either Low Fat or Regular since, we have a new 
item type of NC - so we should re-label the item fat content for the NC's 
as 'non-edible'    

```python
allData.loc[allData['New_Item_Type']=='NC', 
            'Item_Fat_Content'] =\
'Non Edible'
allData['Item_Fat_Content'].value_counts()
```

    Low Fat       6499
    Regular       5019
    Non Edible    2686
    Name: Item_Fat_Content, dtype: int64

The categorical variables need to be replaced with integers so sklearn can 
be used with this data. 

First the categorical values need to be replaced with integers. So if new item 
types are lf r ne, then they will be replaced by 0 1 2. This can be done using 
le.fit_transform 

The get_dummies method from pandas, will create 3 new corresponding dummy 
variables new_item_type_i i-{0,1,2} with 0 or 1 values. 


```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

allData['Outlet'] = le.fit_transform(allData['Outlet_Identifier'])

listOfColumns = ['Item_Fat_Content', 'Outlet_Location_Type',
                 'Outlet_Size', 'New_Item_Type', 'Outlet_Type',
                 'Outlet']

for i in listOfColumns:
    allData[i] = le.fit_transform(allData[i])

# now use one-hot-coding - generate the new categorical variables
allData = pd.get_dummies(allData, columns = listOfColumns)
```

```python
allData.dtypes
```

    Item_Identifier               object
    Item_MRP                     float64
    Item_Outlet_Sales            float64
    Item_Type                     object
    Item_Visibility              float64
    Item_Weight                  float64
    Outlet_Establishment_Year      int64
    Outlet_Identifier             object
    source                        object
    Outlet_Age                     int64
    Item_Fat_Content_0             uint8
    Item_Fat_Content_1             uint8
    Item_Fat_Content_2             uint8
    Outlet_Location_Type_0         uint8
    Outlet_Location_Type_1         uint8
    Outlet_Location_Type_2         uint8
    Outlet_Size_0                  uint8
    Outlet_Size_1                  uint8
    Outlet_Size_2                  uint8
    New_Item_Type_0                uint8
    New_Item_Type_1                uint8
    New_Item_Type_2                uint8
    Outlet_Type_0                  uint8
    Outlet_Type_1                  uint8
    Outlet_Type_2                  uint8
    Outlet_Type_3                  uint8
    Outlet_0                       uint8
    Outlet_1                       uint8
    Outlet_2                       uint8
    Outlet_3                       uint8
    Outlet_4                       uint8
    Outlet_5                       uint8
    Outlet_6                       uint8
    Outlet_7                       uint8
    Outlet_8                       uint8
    Outlet_9                       uint8
    dtype: object

Now we must convert the data back to test and train and save accordingly

```python
# Drop the columns which have been converted to different types:
allData.drop(['Item_Type','Outlet_Establishment_Year'],
             axis=1,inplace=True)

# divide into test and train
newTrainData = allData.loc[allData['source']=='train']
newTestData = allData.loc[allData['source']=='test']

#drop unnecessary column for test data and train data
newTrainData=newTrainData.drop(
    ['source'], axis=1)
newTestData=newTestData.drop(
    ['source', 'Item_Outlet_Sales'], axis=1)

#export as csv
newTrainData.to_csv('train_modified.csv', index=False)
newTestData.to_csv('test_modified.csv', index=False)
```

# 4. Model Building

To start we will create a baseline model to which we can compare our 
subsequent ML models - our baseline will predict the item sales to 
be the average for that item identifier

```python
# get the unique items in the df
uniqueItemList = newTrainData['Item_Identifier'].unique()

#calc an average sales for each item
averageSalesPerItemID = [newTrainData.loc[
                        newTrainData['Item_Identifier']==j, 'Item_Outlet_Sales'].mean()
                         for j in uniqueItemList]

#create a data series for the item ID and average sales
averageSalesPerItemID = pd.Series(averageSalesPerItemID, index=uniqueItemList)
```

```python
baseLine = newTestData.loc[:,['Item_Identifier', 'Outlet_Identifier']]
baseLine['Item_Outlet_Sales'] = baseLine['Item_Identifier'].apply(
                                lambda x: averageSalesPerItemID[x])

baseLine.to_csv('baseLine.csv', index=False)
```

This simple method gets a score of 1599 - we can compare our subsequent ML models 
to this now. Since we will be considering multiple models, its best practice to 
create a function that will take a model, the data, carry out cross validation 
and produce a submission file plus a accuracy score on the test data

```python
from sklearn.model_selection import cross_val_score

def ModelFit(algo, trainSet, testSet, predictors, fileName,
             target='Item_Outlet_Sales', 
             IDcol=['Item_Identifier', 'Outlet_Identifier']):
    
    # first fit the algo on the data
    algo.fit(trainSet[predictors], trainSet[target])
    
    # make predictions on the training Set
    train_predictions = algo.predict(trainSet[predictors])
    
    # perform cross-validation 
    # this is splitting training data to train the model and test it 
    # then we can iterate over the combinations of the split and take the 
    # parameters at the end as an average
    cv_score = cross_val_score(algo,
                               trainSet[predictors], trainSet[target],
                               cv=10)
    print(cv_score)
    
    # print model report
    print('\n', 'Model Report')
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
    
    # make predictions on the test data
    testPredict = algo.predict(testSet[predictors])
    
    # export submission file
    subm = testSet.loc[:,IDcol]
    subm[target] = testPredict
    subm.to_csv(fileName, index=False)
        
```

## Linear Regression

Our first attempt will be the simplest!

```python
target='Item_Outlet_Sales'
IDcol=['Item_Identifier', 'Outlet_Identifier']
```


```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

predictors = [x for x in newTrainData.columns
              if x not in [target]+IDcol]

algo_LR = LinearRegression(normalize=True)
ModelFit(algo_LR, newTrainData, newTestData, predictors, 'LinearReg.csv')

coef = pd.Series(algo_LR.coef_,predictors).sort_values()
plt.figure(figsize=(7,7))
coef.plot.bar()
plt.show()
```

    [ 0.56103237  0.58289587  0.5432337   0.56826986  0.50917172  0.57692204
      0.57852675  0.55512497  0.58038753  0.54801261]
    
     Model Report
    Accuracy: 0.56 (+/- 0.04)

![png](/images/MLPipeline-BigMartSales/IntroToML_56_1.png)

the score of the above linear regression model is 1203 after submission

some coefficients are very large - try regularisation now, via ridge regression

```python
algo_LR_ridge = Ridge(alpha=0.1, normalize=True)
ModelFit(algo_LR_ridge, newTrainData, newTestData, predictors, 'LinearReg_ridge.csv')
coef = pd.Series(algo_LR_ridge.coef_,predictors).sort_values()
plt.figure(figsize=(7,7))
coef.plot.bar()
plt.show()
```

    [ 0.5541056   0.57638477  0.54039697  0.56654862  0.51818684  0.5729066
      0.57355431  0.5503474   0.57721427  0.5469673 ]
    
     Model Report
    Accuracy: 0.56 (+/- 0.04)

![png](/images/MLPipeline-BigMartSales/IntroToML_59_1.png)

The above submission gives a score of 1203 too so let's try a different method

## Decision Trees

```python
from sklearn.tree import DecisionTreeRegressor
predictors = [x for x in newTrainData.columns
              if x not in [target]+IDcol]
algo_dt = DecisionTreeRegressor(max_depth=30, min_samples_leaf=100)
ModelFit(algo_dt, newTrainData, newTestData, predictors, 'dt.csv')
features = pd.Series(algo_dt.feature_importances_,predictors).sort_values()
plt.figure(figsize=(7,7))
features.plot.bar()
plt.show()
```

    [ 0.5951301   0.61033125  0.55507546  0.60450386  0.53260197  0.60463126
      0.6091992   0.5911486   0.62400752  0.55901902]
    
     Model Report
    Accuracy: 0.59 (+/- 0.06)

![png](/images/MLPipeline-BigMartSales/IntroToML_62_1.png)

```python
# Take the 5 most important features and use them only in another decision tree

predictors = ['Item_MRP', 'Outlet_Type_0', 'Outlet_Type_3', 
              'Outlet_Age', 'Item_Visibility', 'Item_Weight']

algo_dt_specific = DecisionTreeRegressor(max_depth=30, min_samples_leaf=50)
ModelFit(algo_dt_specific, newTrainData, newTestData, predictors, 'dt_imp.csv')
features = pd.Series(algo_dt_specific.feature_importances_,predictors).sort_values()
plt.figure(figsize=(7,7))
features.plot.bar()
plt.show()
```

    [ 0.58294749  0.60826938  0.53674156  0.57975795  0.50841877  0.60075873
      0.60545207  0.59032418  0.62020541  0.5445636 ]
    
     Model Report
    Accuracy: 0.58 (+/- 0.07)

![png](/images/MLPipeline-BigMartSales/IntroToML_63_1.png)

Above only gets 1169

```python
#let's try a random forest 
from sklearn.ensemble import RandomForestRegressor

algo_rf = RandomForestRegressor(n_estimators=400,max_depth=6, 
                                min_samples_leaf=100,n_jobs=4)
ModelFit(algo_rf, newTrainData, newTestData, predictors, 'rf.csv')
features = pd.Series(algo_dt_specific.feature_importances_,predictors).sort_values()
plt.figure(figsize=(7,7))
features.plot.bar()
plt.show()
```

    [ 0.60103919  0.60965552  0.565355    0.60705907  0.5418581   0.60294301
      0.62525588  0.59259296  0.62700022  0.57154247]
    
     Model Report
    Accuracy: 0.59 (+/- 0.05)

![png](/images/MLPipeline-BigMartSales/IntroToML_65_1.png)

We could continue to try to play with the parameters but rather than that let's take 
a re-look at the data using seaborn and see if we can use gradient boosting to increase
out accuracy to more than 60%
