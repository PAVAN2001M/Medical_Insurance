#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import linear_model
import math
import seaborn as sn
import matplotlib.pyplot as plt
import bottle
from bottle import route, run, request



# In[3]:


df=pd.read_csv(r'C:\Users\Apoorva\Desktop\insurance3r2.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.shape


# In[9]:


x=df.drop(columns=['charges'])
Y=df['charges']


# In[11]:


X=sm.add_constant(x)


# In[14]:


X


# In[15]:


Y


# In[12]:


X_feat=x.columns


# In[16]:


X_feat


# In[17]:

#train test split
train_X, test_X, train_y, test_y = train_test_split(X, Y,train_size = 0.8, random_state = 42 )


# In[18]:


medical_model_1 = sm.OLS(train_y, train_X).fit()
medical_model_1.summary2()


# In[19]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors( X ):
    #X_matrix = X.as_matrix().values
    vif_factors = pd.DataFrame()
    vif = [ variance_inflation_factor( X.values, i ) for i in range( X.shape[1] ) ]
   
    vif_factors['column'] = X.columns
    vif_factors['VIF'] = vif
    return vif_factors

vif_fact = pd.DataFrame()
vif_fact = get_vif_factors( X[X_feat] )
vif_fact


# In[20]:


columns_with_large_vif = vif_fact[vif_fact.VIF > 4].column
plt.figure( figsize = (16,10) )
sn.heatmap( X[columns_with_large_vif].corr(), annot = True )
plt.title("Heatmap depicting correlation between features")


# In[21]:


column_removed=['age','bmi']
X_new_features=list(set(X_feat)-set(column_removed))
get_vif_factors(X[X_new_features])


# In[24]:


train_X=train_X[X_new_features]
medical_model_2=sm.OLS(train_y,train_X).fit()
medical_model_2.summary2()


# In[25]:


significant_vars=['steps','smoker','region','children','sex']
train_X=train_X[significant_vars]


# In[26]:


medical_model_3=sm.OLS(train_y,train_X).fit()


# In[28]:


medical_model_3.summary2()


# In[30]:


pred_y=medical_model_3.predict(test_X[significant_vars])
from sklearn.metrics import r2_score
np.round(r2_score(pred_y,test_y),2)


# In[31]:


train_y=np.sqrt(train_y)
medical_model_4=sm.OLS(train_y,train_X).fit()


# In[32]:


medical_model_4.summary2()


# In[33]:


pred_y=np.power(medical_model_4.predict(test_X[train_X.columns]),2)
np.round(metrics.r2_score(pred_y,test_y),2)


# In[34]:


from sklearn import metrics
np.sqrt(metrics.mean_squared_error(pred_y,test_y))


# In[35]:


pr=pd.DataFrame({'Actual': test_y, 'Predicted': pred_y})
pr

 #HTML form for input
input_form = """
<!DOCTYPE html>
<html>
<head>
    <title>Air Quality Index Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        h2 {
            text-align: center;
        }
        form {
            width: 50%;
            margin: 0 auto;
        }
        label {
            display: block;
            margin-bottom: 5px;
        }
        input[type="text"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            text-align: center;
        }
    </style>
</head>
<body>

<h2>Medical Insurance Prediction</h2>

<form action="/predict" method="post">
  <label for="steps">Steps:</label>
  <input type="text" id="steps" name="steps" required><br>
  <label for="smoker">Smoker:</label>
  <input type="text" id="smoker" name="smoker" required><br>
  <label for="region">Region:</label>
  <input type="text" id="region" name="region" required><br>
  <label for="children">Children:</label>
  <input type="text" id="children" name="children" required><br>
  <label for="sex">sex:</label>
  <input type="text" id="sex" name="sex" required><br>
  <input type="submit" value="Predict">
</form>

</body>
</html>
"""

# Initialize Bottle app
app = bottle.default_app()

# Route for home page
@app.route('/')
def home():
    return input_form

# Route for prediction
@app.route('/predict', method='POST')
def prediction():
    steps= float(request.forms.get('steps'))
    smoker= float(request.forms.get('smoker'))
    region= float(request.forms.get('region'))
    children= float(request.forms.get('children'))
    sex = float(request.forms.get('sex'))
    
    input_data = sm.add_constant(np.array([[steps,smoker,region,children,sex]]))
    prediction_result = medical_model_4.predict(input_data)[0]
    
    return "Predicted Insurance: {}".format(prediction_result)

if __name__ == '__main__':
    run(host='localhost', port=8080)


# In[ ]:





# In[ ]:




