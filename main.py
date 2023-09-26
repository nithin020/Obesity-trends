# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.preprocessing import OneHotEncoder
#Loading the dataset into a Pandas DataFrame

csv_file_path = r"C:/Users/LENOVO/Documents/Python Projects/Obesity Trends/obesity.csv"

df = pd.read_csv(csv_file_path)
df
df.info()
# Summary of the data
#Dropping the unnecessary columns in data
df = df.drop(['Low_Confidence_Limit', 'High_Confidence_Limit', 'Total', 'Data_Value_Alt', 'YearEnd', 'Topic', 'Class', 'Datasource', 'Data_Value_Unit', 'QuestionID', 'ClassID', 'TopicID', 'DataValueTypeID', 'Data_Value_Type', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote', 'StratificationCategoryId1', 'StratificationID1'], axis=1)
df = df.drop(['GeoLocation', 'Question', 'StratificationCategory1', 'Stratification1', 'Gender'], axis=1)

df.info()
#Dropping the unnecessary columns in data

#subseting the data to be able to analyse according to specific column.
#df=df.dropna(subset=['Income'])
df=df.dropna(subset=['Education'])
df.info()
#Standardizing the column names
#creating a function to be able to lower all letters in columns
def lower_case_column_names(df):
    df.columns=[i.lower() for i in df.columns]
    return df
df=lower_case_column_names(df)
df
#checking the data types to see if any correction needed
df.dtypes
#Checking for duplicate & missing values
#Checking duplicate variable
print(df.duplicated().sum())
#dropping duplicate data
df = df.drop_duplicates()
df.isna()
#checking for any high percentage of missing value in columns
percent_missing = df.isnull().sum() * 100 / len(df)
percent_missing
#dropping columns with high volume of missing values
df = df.drop(['age(years)'],axis=1)
#Going by Alex's & Peter's new advice we would drop age, education, gender, income and race/ethnicity.
#So we can compare the
#Removing outliers
#creating function to remove outliers by using interquartile function
def remove_outliers(df):
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        df.drop(df[(df[col] < lower_bound) | (df[col] > upper_bound)].index, inplace=True)
remove_outliers(df)
df
#Standardizing the text in the data
#Creating a function for standardizing the string data by lowering all the letters
def standardize_text(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.lower()
standardize_text(df)
df
#Data split for Numerical and Categorical variables
numerical_columns = df.select_dtypes(include=['int', 'float']).columns
numerical = df[numerical_columns]
numerical
numerical = numerical.fillna(numerical.mean())
numerical
categorical_columns = df.select_dtypes(include=['object']).columns
categorical = df[categorical_columns]
categorical
categorical
#One-Hot Encoding for Categorical variables
categorical.columns
#income
in_map = {'less than $15,000':0, '$15,000 - $24,999':1, '$25,000 - $34,999':2,'$35,000 - $49,999':3, '$50,000 - $74,999':4, 
          '$75,000 or greater':5, 'data not reported':6}
in_map
df['income'] = df['income'].map(in_map)
#education
ed_map = {'less than high school': 0, 'high school graduate':1, 'some college or technical school':2,'college graduate':3}
ed_map
df['education'] = df['education'].map(ed_map)
#race/ethnicity
rc_map = {'non-hispanic white': 0, 'non-hispanic black':1, 'hispanic':2,'asian':3, 'hawaiian/pacific islander':4,
       'american indian/alaska native':5, '2 or more races':6, 'other':7}
rc_map
df['race/ethnicity'] = df['race/ethnicity'].map(rc_map)
df
#Ploting distributions of numerical data
for col in numerical.columns:
    sns.displot(numerical[col])
    plt.show()
numerical.info()
numerical.hist(figsize=(20,15), grid=False)
plt.show()
#Plot relationships between Numerical variables
sns.pairplot(numerical)
#checking correlation between numerical variables
numerical.corr()
corr = numerical.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, 
        annot=True, cmap='Reds')
#X/Y Split
#X = df[['yearstart','income', 'locationid']]
X = df[['yearstart','education', 'locationid']]
y = numerical['data_value']
#Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.head(3)
std_scaler=StandardScaler().fit(X_train) 

X_train_scaled=std_scaler.transform(X_train)
X_test_scaled=std_scaler.transform(X_test)
print(X_train_scaled)
print("--------")
print(X_test_scaled)
#Creating the Model
#creating OLS model
X_train_const_scaled = sm.add_constant(X_train_scaled) # adding a constant

model = sm.OLS(y_train, X_train_const_scaled).fit()
predictions_train = model.predict(X_train_const_scaled) 

X_test_const_scaled = sm.add_constant(X_test_scaled) # adding a constant
predictions_test = model.predict(X_test_const_scaled) 
print_model = model.summary()
print(print_model)
predictions_test
#display adjusted R-squared
print(model.rsquared_adj)
#Model Fitting
#creating linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
#Model Parameters
model.coef_
model.intercept_
model.score(X_test_scaled,y_test)
#Model Predictions
y_pred=model.predict(X_test_scaled)  
y_test
y_pred
result=pd.DataFrame({"y_test":y_test,"y_pred":y_pred})

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Scatter plot of y_pred vs y_test
axs[0, 0].plot(y_pred, y_test, 'o')
axs[0, 0].set_xlabel("y_test")
axs[0, 0].set_ylabel("y_pred")
axs[0, 0].set_title("Test Set - Predicted vs Real")

# Plot 2: Histogram of residuals
axs[0, 1].hist(y_test - y_pred)
axs[0, 1].set_xlabel("Test y - y_pred")
axs[0, 1].set_title("Test Set Residual Histogram")

# Plot 3: Scatter plot of y_pred vs residuals
axs[1, 0].plot(y_pred, y_test - y_pred, "o")
axs[1, 0].set_xlabel("Predicted")
axs[1, 0].set_ylabel("Residuals")
axs[1, 0].set_title("Residuals by Predicted")

# Plot 4: Regression plot using Seaborn
sns.regplot(x='y_test', y='y_pred', data=result, scatter_kws={"color": "red"}, line_kws={"color": "black"}, ax=axs[1, 1])

# Adjust layout to prevent overlapping
plt.tight_layout()

# Evaluate the model
mse_value = mse(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
print("Mean Squared Error:", mse_value)
print("R-squared:", r2)
print("Adjusted R-squared:", adj_r2)

# Feature Importance
features_importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': abs(model.coef_)
})
features_importances = features_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(x=features_importances['Attribute'], height=features_importances['Importance'], color='#087E8B')
plt.title('Feature Importances Obtained from Coefficients', size=15)
plt.xticks(rotation='vertical')
plt.show()#saving cleaned data
df.to_csv("C:/Users/LENOVO/Documents/Python Projects/Obesity Trends/obesity_clean.csv")
