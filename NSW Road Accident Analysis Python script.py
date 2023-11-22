#!/usr/bin/env python
# coding: utf-8

# In[73]:


#Load essential libraries for processing and visualising data 

import numpy as np 
import pandas as pd               #Processing data
import matplotlib.pyplot as plt   #Visualisation 
import seaborn as sns             #Visualisation


# In[74]:


df = pd.read_excel('NSW_Road_Crash_Data_2017-2021_CRASH.xlsx')
df.head()


# In[4]:


df.info()


# ## Data Cleaning

# In[84]:


#Categorical columns 
cat_cols = df.select_dtypes(include=['object', 'category']).columns
cat_cols


# In[85]:


#Replace null values in categorical columns with the mode 
df[cat_cols] = df[cat_cols].apply(lambda x: x.fillna(x.mode()[0]))

print(df.isnull().sum())


# In[86]:


#Numerical_columns 
numerical_cols = df.select_dtypes(include=['float64', 'int']).columns
numerical_cols


# In[87]:


#Replace null values in numerical columns with the mean 
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.mean()))

print(df.isnull().sum())


# In[88]:


df.duplicated().sum() #Check duplicates 


# ## Visualisation

# ### Bar chart - top five towns/suburbs with highest no. of crashes

# In[15]:


top_10_towns = df.groupby('Town')['CrashID'].count().sort_values(ascending=False).head(10) #Select 10 towns with highest accident/crash numbers 
top_10_towns_reorder = top_10_towns.sort_values(ascending=True)  #order them in ascending order 

plt.figure(figsize=(12, 6)) #fix the visual size 

sns.barplot(x=top_10_towns_reorder.values, y=top_10_towns_reorder.index, orient='h')  #seaborn horizontal bar plot

plt.xlabel('Crash Count')
plt.ylabel('Town')
plt.title('Top Ten Towns With Highest Crash Counts')

plt.show()


# ### Pie chart - display the proportion of fatalities occurring in different weathers

# In[22]:


plt.style.use('classic')
plt.style.use('default')

# Filter out data from the year 2016
df1 = df[df['Reporting_year'] != 2016]

# Filter out Unknown weather
df1 = df1[df1['Weather'] != 'Unknown']
df1 = df1[df1['Weather'] != 'Other']


# Remove duplicates based on specific columns
d1 = df1.drop_duplicates(subset=['CrashID'])

plt.figure(figsize=(12, 6))
df1.groupby('Weather').sum().plot(kind='pie', 
                                                title ='Distribution of fatalities by weather conditions',
                                                y='No_killed',  legend = False,
                                                ylabel = '', autopct='%1.1f%%')

plt.figure(figsize=(12, 6)) #fix the visual size 

plt.show()


# ### Line plot & bar chart - distribution of the number of people killed on the road vary by month & year

# In[23]:


#create new dataframe for grouped data
grouped_data = df.groupby(['Year_of_crash', 'Month_of_crash'])['No_killed'].sum()
grouped_data = grouped_data.reset_index()

#Clean out rows where the year is 2016 or 0
grouped_data['Year_of_crash'] = grouped_data['Year_of_crash'].fillna(0).astype(int)
grouped_data = grouped_data[(grouped_data['Year_of_crash'] != 2016) & (grouped_data['Year_of_crash'] != 0)]

#creating format for x axis
grouped_data['Year-Month'] = grouped_data['Year_of_crash'].astype(str).str[:4] + '-' + grouped_data['Month_of_crash'].str[:3]

#map out the months using a dictionary to order it later
month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
grouped_data['Month_Number'] = grouped_data['Month_of_crash'].map(month_dict)

#Sort the data by the 'Year and Month_Number' column
grouped_data = grouped_data.sort_values(['Year_of_crash', 'Month_Number'])

#creating line
plt.figure(figsize=(12, 6))
plt.plot(grouped_data['Year-Month'], grouped_data['No_killed'], color = 'r')

#creating bar
plt.bar(grouped_data['Year-Month'], grouped_data['No_killed'], width=0.5)
plt.xticks(rotation = 90)
plt.xlabel("Year - Month")
plt.ylabel("Number of people killed")
plt.title("Number of people killed in between 2017 and 2021")
plt.show()


# ### Line plot & bar chart - show the trend in the number of fatalities and injuries over years

# In[25]:


df['Year_of_crash'] = df['Year_of_crash'].astype(int)
#change dtype of year_of_crash column
df.dtypes #check if the dtype changed

#Total fatalities over year 
year_total_fatal = df.groupby('Year_of_crash')['No_killed'].sum().reset_index()
year_total_fatal = year_total_fatal[(year_total_fatal['Year_of_crash'] != 0) & (year_total_fatal['Year_of_crash'] != 2016)]   #remove year 2016 and 0 
year_total_fatal

#Total injuries over year 
year_total_injury = df.groupby('Year_of_crash')[['No_seriously_injured', 'No_moderately_injured', 'No_minor_other_injured']].sum().reset_index() #Sum of each injury type
year_total_injury['Total_Injuries'] = year_total_injury[['No_seriously_injured', 'No_moderately_injured', 'No_minor_other_injured']].sum(axis=1)  #Total injuries of all 3 types
year_total_injury = year_total_injury[(year_total_injury['Year_of_crash'] != 0) & (year_total_injury['Year_of_crash'] != 2016)]   #remove year 2016 and - 
year_total_injury

import matplotlib.pyplot as plt  #import visualisation library

year = year_total_fatal['Year_of_crash']  #select values from year_of_crash column in year_total_fatal dataframe for plotting

total_injuries = year_total_injury['Total_Injuries']   #select values from total_injuries column in year_total_fatal dataframe for plotting

total_fatalities = year_total_fatal['No_killed']   #select values from no_killed column in year_total_fatal dataframe for plotting

plt.figure(figsize=(12,6))  #define plot size 
fig, ax1 = plt.subplots()


ax1.set_xlabel('Year')

#Create line plot for total fatalities --- total fatalities on the left y-axis
ax1.set_ylabel('Fatalities', color='black')
ax1.plot(year, total_fatalities, marker='o', color = 'red', label='Fatalities')

#Create barplot for total injuries -- total injuries on the right y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Injuries', color='black')
ax2.bar(year, total_injuries, color = 'blue', alpha = 0.2, label='Injuries')

# Display the fatality numbers 
for i,j in zip(year, total_fatalities):
    ax1.annotate(str(j),xy=(i,j))

plt.title("Trends in Fatalities and Injuries Over the Years")


# Add a single legend for both bar and line plot
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines + lines2
labels = labels + labels2

# Display the combined legend in the upper right
ax2.legend(lines, labels, loc = 'upper right')

#Show the plot
plt.show()


# ### Stacked bar chart - relationship between the speed limit and accident statistics

# In[26]:


# Define the desired order of speed limits
speed_limit_order = [
    '10 km/h', '20 km/h', '30 km/h',
    '40 km/h', '50 km/h', '60 km/h',
    '70 km/h', '80 km/h', '90 km/h',
    '100 km/h', '110 km/h'
]

# Convert 'Speed_limit' to categorical with the desired order
df['Speed_limit'] = pd.Categorical(
    df['Speed_limit'], categories=speed_limit_order, ordered=True)


# Group the data by 'Speed_limit' and 'Degree_of_crash' and count occurrences
grouped_data = df.groupby(['Speed_limit', 'Degree_of_crash']).size().unstack()

# Create a stacked bar chart
ax = grouped_data.plot(kind='bar', stacked=True, figsize=(12, 6), alpha=0.7)  # Adjust alpha for visibility

# Set labels and title
plt.xlabel("Speed Limit")
plt.ylabel("Count")
plt.title("Stacked Bar Plot of Degree of Crash by Speed Limit")

# Show the legend
plt.legend(title="Degree of Crash")

# Rotate x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Show the plot
plt.show()


# ### Bar chart - show most notorious locations in the past 5 years

# In[28]:


df6 = pd.DataFrame()
df6['Town'] = df['Town']
df6['Street_of_crash'] = df['Street_of_crash']
df6['Street_type'] = df['Street_type']
df6['Year_of_crash'] = df['Year_of_crash']

#Concat columns
df6['Concatenated'] = df6['Town'] + '-' + df6['Street_of_crash'] + ' ' + df6['Street_type']

#Clean out rows where the year is 2016 or 0
df6['Year_of_crash'] = df6['Year_of_crash'].astype(int)
df6 = df6[(df6['Year_of_crash'] != 2016) & (df6['Year_of_crash'] != 0)]

#Group by 'Concatenated' and count the number of occurrences
grouped_df6 = df6.groupby('Concatenated').size().reset_index(name='count')

#Select the top 10 locations
grouped_df6 = grouped_df6.sort_values('count', ascending=True).tail(10)  

#Create a horizontal bar chart
plt.figure(figsize=(10, 6))
plt.barh(grouped_df6['Concatenated'], grouped_df6['count'])
plt.xlabel('Count')
plt.title('Worst location by Count')
plt.show()


# ### Density heatmap - show crashes in geographical locations

# In[30]:


pip install dash  #Install dash library


# In[31]:


import plotly.express as px  #import plotly package 


# In[33]:


df7 = pd.DataFrame()
df7['Magnitude'] = df['No_killed'] + df['No_seriously_injured']
df7['Longitude'] = df['Longitude']
df7['Latitude'] = df['Latitude']
df7['Year_of_crash'] = df['Year_of_crash']

#Change data type for column year of crash
df7['Year_of_crash'] = df7['Year_of_crash'].astype(int)

#Clean out rows where the year is 2016 or 0
df7 = df7[(df7['Year_of_crash'] != 2016) & (df7['Year_of_crash'] != 0)]

#removes Magnitude where it is 0
df7 = df7[df7['Magnitude'] != 0]

#sets limits
df7['Magnitude'] = df7['Magnitude'].clip(1, 8)

#creating density heatmap
fig = px.density_mapbox(df7, lat='Latitude', lon='Longitude', z='Magnitude', radius=10, center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain", animation_frame="Year_of_crash", range_color=[1, 8])
fig.show()


# ### Heatmap to observe the correlation between variables 

# In[34]:


df_cleaned = df.drop(columns=['DCA_code', 'RUM_code'])
fig, ax = plt.subplots(figsize=(10,10))
cm = sns.heatmap(df_cleaned.corr(), linewidths = .5, cmap="YlGnBu", annot=True, ax=ax, fmt='.1g')


# ## Predictive Modelling

# ### Linear Regression - predict severity of crashes from speed limit

# In[43]:


df.dtypes


# In[50]:


#Preparing data for modelling 

#df['Year_of_crash'] = df['Year_of_crash'].fillna(0).astype(int)
#df['Reporting_year'] = df['Reporting_year'].fillna(0).astype(int)
df = df[(df['Year_of_crash'] != 2016) & (df['Year_of_crash'] != 0)]
df = df[(df['Reporting_year'] != 2016) & (df['Reporting_year'] != 0)]

df_data = df[['Speed_limit','No_of_traffic_units_involved', 'No_killed', 'No_seriously_injured', 'No_moderately_injured', 'No_minor_other_injured']].copy()

#df_data.columns[df_data.isna().any()].tolist()

df_data = df_data[df_data['Speed_limit'] != 'Unknown'].copy()
df_data['Speed_limit'].unique()

#df_data.head()

df_data['Speed_limit'] = df_data['Speed_limit'].str.extract('(\d+)').fillna(0).astype(int)

# Add the Crash_Severity column to the DataFrame
df_data['Crash_Severity'] = (
    df_data['No_killed'] * 16 +
    df_data['No_seriously_injured'] * 8 +
    df_data['No_moderately_injured'] * 4 +
    df_data['No_minor_other_injured'] * 2 +
    df_data['No_of_traffic_units_involved'] * 1
)


# In[51]:


#Modelling Regression model

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Extract features and target variable
X = df_data[['Speed_limit']]
y = df_data['Crash_Severity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

plt.figure(figsize=(10, 6))
sns.boxplot(x='Speed_limit', y='Crash_Severity', data=df_data)
plt.title('Crash Severity Distribution for Different Speed Limits')
plt.show()

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

print(f'Coefficient: {model.coef_[0]}')
print(f'Intercept: {model.intercept_}')

import matplotlib.pyplot as plt

# Plot the data points
plt.scatter(X_test, y_test, color='black', label='Data', s=5)

# Plot the linear regression line
plt.plot(X_test, y_pred, color='blue', linewidth=1, label='Linear Regression Line')

# Labeling the plot
plt.title('Linear Regression: Speed Limit vs. Crash Severity')
plt.xlabel('Speed Limit (km/h)')
plt.ylabel('Crash Severity')
plt.legend()

# Show the plot
plt.show()


# ### Decision Tree - improve model; predict severity of crashes based on speed limit

# In[53]:


#Preparing data for modelling decision tree 

#convert year to integers
#df['Year_of_crash'] = df['Year_of_crash'].fillna(0).astype(int)
#df['Reporting_year'] = df['Reporting_year'].fillna(0).astype(int)

#exclude 2016
#df = df[(df['Year_of_crash'] != 2016) & (df['Year_of_crash'] != 0)]
#df = df[(df['Reporting_year'] != 2016) & (df['Reporting_year'] != 0)]


#Select columns
df_data = df[['Degree_of_crash_detailed','Speed_limit']].copy()

df_data.columns[df_data.isna().any()].tolist()

#Remove unknown from speed limit
df_data = df_data[df_data['Speed_limit'] != 'Unknown'].copy()
df_data['Speed_limit'].unique()

df_data.head()

#Convert string into integer, extracting number
df_data['Speed_limit'] = df_data['Speed_limit'].str.extract('(\d+)').fillna(0).astype(int)

df_data.head()


# In[54]:


#Decision tree modelling 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix

#Split data
X = df_data[['Speed_limit']]
y = df_data['Degree_of_crash_detailed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)

#Train model
model.fit(X_train, y_train)

#Make prediction
y_pred = model.predict(X_test)

#Print report
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

#Visualise tree
plt.figure(figsize=(15, 8))
plot_tree(model, filled=True, feature_names=['Speed_limit'], class_names=None)
plt.show()


# ### Linear Regression - Severity of crash vs. Day of week 

# In[57]:


df_no_nan = df.dropna(subset=['Degree_of_crash', 'No_killed', 'Day_of_week_of_crash'])

from sklearn import linear_model

#clarifying each category into numerical values
def crash_severity(value):
    if value == "Fatal":
        return 5
    elif value == "Injury":
        return 3
    else:
        return 1
#assigning numerical values to days of week
def weekdayno(value):
    if value == "Monday":
        return 1
    elif value == "Tuesday":
        return 2
    elif value == "Wednesday":
        return 3
    elif value == "Thursday":
        return 4
    elif value == "Friday":
        return 5
    elif value == "Saturday":
        return 6
    elif value == "Sunday":
        return 7
    


df_no_nan['weekdayno'] = df_no_nan['Day_of_week_of_crash'].map(weekdayno)
df_no_nan['crash_severity'] = df_no_nan['Degree_of_crash'].map(crash_severity)


#df_no_nan.head()

regr = linear_model.LinearRegression()

severity= df_no_nan[['crash_severity']]
weekdayno= df_no_nan[['weekdayno']]

regr.fit(weekdayno, severity)

plt.scatter( df_no_nan.weekdayno, df_no_nan.crash_severity, color='Black')

plt.plot(weekdayno, regr.coef_[0][0]*weekdayno + regr.intercept_[0], '-r')
plt.xlabel("weekday")
plt.ylabel("crash severity")
plt.title('crash severity based on day of week', fontsize=14)


# ### Logistic Regression - predict the likelihood of a crash being severe based on features like weather, road alignment, and road classification

# In[75]:


#Import important libraries 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# In[77]:


#Step 1 - Separate x and y data 
from sklearn.preprocessing import LabelEncoder


X1 = df[['Weather', 'Alignment', 'Road_classification']]  #predictor variables
print(X1)

#Because the logistic reg only accepts categorical or numerical independent variables 
X1 = pd.get_dummies(X1, columns=['Weather', 'Alignment', 'Road_classification'])

#Understanding severity of crashes 
#road_accident['Degree_of_crash'].value_counts()

#There are 3 types of crash outcomes 
#In order to conduct a logistic regression, these outcomes must be
#categorized into binary variables 
#Decision: map fatal to value 1, Injury & Towaway to value 0 


#Create binary variables to the crash severity variable 
df['Crash_severe'] = df['Degree_of_crash'].map({'Fatal':1, 'Injury':1, 'Non-casualty (towaway)':0})
y1 = df[['Crash_severe']]
print(y1)

#Step 2 - Training and testing set split 

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)


#After this, use resampling method: undersampling 


# In[78]:


pip install -U imbalanced-learn   #Import package for undersampling  


# In[79]:


#Under-resampling technique 

from imblearn.under_sampling import RandomUnderSampler # Import the necessary libraries

rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')

# Balancing the data
X_resampled, y_resampled = rus.fit_resample(X1_train, y1_train)


# In[81]:


#Step 3 - build logistic regression model
r_accident_lr = LogisticRegression(solver = 'liblinear', random_state = 42, C = 10) 

#Step 4 - Fit the model to the resampled data
r_accident_lr.fit(X_resampled, y_resampled)


#Step 5 - Predict the test data

y1_pred = r_accident_lr.predict(X1_test)



print(pd.DataFrame(y1_pred))

#Step 6 - Evaluate test the model 
r_accident_lr.score(X1_test, y1_test)

#Evaluation using confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_visual(y1_true, y1_pred, labels):
    cm = confusion_matrix(y1_true, y1_pred)
    
    plt.figure(figsize=(6, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

confusion_matrix_visual(y1_test, y1_pred, ['low', 'high'])

#Precision report 
from sklearn.metrics import classification_report

print("Report : ",
    classification_report(y1_test, y1_pred))

# ROC Curves
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y1_test, r_accident_lr.predict_proba(X1_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()


# In[82]:


# Precision-Recall Curves
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y1_test, r_accident_lr.predict_proba(X1_test)[:, 1])

# Calculate AUC for precision-recall curve
pr_auc = auc(recall, precision)

# Plot precision-recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='red', lw=2, label='PR curve (area = %0.2f)' % pr_auc)
plt.axhline(y=y1_test.mean().iloc[0], color='blue', linestyle='--', label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc="lower left")
plt.show()


# ### Feature selection for predictive models - using Chi-square method

# In[89]:


import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder


# Convert categorical variables to numerical labels using LabelEncoder
label_encoder = LabelEncoder()
encoded_data = df.copy()
categorical_columns = ['Month_of_crash', 'Day_of_week_of_crash', 'Two_hour_intervals', 'Street_of_crash', 'Street_type', 'Direction', 'Identifying_feature', 'Identifying_feature_type', 'Town', 'School_zone_location', 'School_zone_active', 'Type_of_location', 'LGA', 'Urbanisation', 'Alignment', 'Primary_permanent_feature', 'Primary_temporary_feature', 'Primary_hazardous_feature', 'Street_lighting', 'Road_surface', 'Surface_condition', 'Weather', 'Natural_lighting', 'Signals_operation', 'Other_traffic_control', 'Speed_limit', 'Road_classification']

for col in categorical_columns:
    encoded_data[col] = label_encoder.fit_transform(df[col])

# Create a DataFrame with only the relevant columns
X_categorical = encoded_data[categorical_columns]
y = encoded_data['Degree_of_crash']

# Apply chi-squared test
chi2_stat, p_values = chi2(X_categorical, y)

# Create a DataFrame to display the results
chi2_results = pd.DataFrame({
    'Feature': categorical_columns,
    'Chi2 Statistic': chi2_stat,
    'P-Value': p_values
})

# Display the results
print(chi2_results)


# ### Logistic Regression - predict severity of crashes based on features with high chi2 stats: street of crash, type of location, street lighting

# In[91]:


#Preparing data for modelling
X2 = df[['Street_of_crash', 'Street_lighting', 'Type_of_location']]
X2 = pd.get_dummies(X2, columns=['Street_of_crash', 'Street_lighting', 'Type_of_location'])


#Create binary variables to the crash severity variable 
df['Crash_severe'] = df['Degree_of_crash'].map({'Fatal':1, 'Injury':1, 'Non-casualty (towaway)':0})
y2 = df[['Crash_severe']]
print(y2)


#Splitting training and testing dataset
from sklearn.model_selection import train_test_split # Import train_test_split function

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=1) # 70% training and 30% test


# In[92]:


pip install -U imbalanced-learn   #Import package for undersampling 


# In[93]:


from imblearn.under_sampling import RandomUnderSampler # Import the necessary libraries
rus = RandomUnderSampler(random_state=42, sampling_strategy = 'majority')
# Balancing the data
X2_resampled, y2_resampled = rus.fit_resample(X2_train, y2_train)
#Build model
r_accident_lr = LogisticRegression(solver = 'liblinear', random_state = 42, C = 10) 


# In[94]:


#Step 3: Fit the model with data 

r_accident_lr.fit(X2_train, y2_train) 


#Step 4: Predicting 
y2_pred = r_accident_lr.predict(X2_test)

print(pd.DataFrame(y2_pred).value_counts())



#Step 5 - Evaluate test the model 
accuracy = r_accident_lr.score(X2_test, y2_test)
print(accuracy)

#Confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix_visual(y2_true, y2_pred, labels):
    cm = confusion_matrix(y2_true, y2_pred)
    
    plt.figure(figsize=(6, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

confusion_matrix_visual(y2_test, y2_pred, ['low', 'high'])




#Precision report 
from sklearn.metrics import classification_report

print("Report : ",
    classification_report(y2_test, y2_pred))

# ROC Curves
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y2_test, r_accident_lr.predict_proba(X2_test)[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

