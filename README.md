<H3>JWALAMUKHI S</H3>
<H3>212223040079</H3>
<H3>EX. NO.1</H3>
<H3>DATE</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
print("datahead")
print(data.head())
print("datatail")
print(data.tail())

X=data.iloc[:,:-1].values
print("X")
print(X)

y=data.iloc[:,-1].values
print("y")
print(y)
print("datainfo")
data.info()

print("Missing Values: \n ",data.isnull().sum())

print("Duplicate values:\n ")
print(data.duplicated())

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)

data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print("Normalized data \n" , df1)

X = data.drop('Exited', axis=1)  
y = data['Exited'] 

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("Training data")
print(X_train)
print(y_train)

print("Testing data")
print(X_test)
print(y_test)
print("Length of X_test: ", len(X_test))
```


## OUTPUT:

![exp1-1](https://github.com/user-attachments/assets/436a61d1-482b-45c7-b12b-a5ad6e6f9481)

![exp1-2](https://github.com/user-attachments/assets/20c1c397-895f-48e6-b130-8f9164917cc3)

![exp1-3](https://github.com/user-attachments/assets/6ca30d01-a453-4785-83b8-6d9a462c45ed)

![exp1-4](https://github.com/user-attachments/assets/05dfb59a-9dfe-4a7d-a1c3-20fd1b25e9db)


![exp1-5](https://github.com/user-attachments/assets/3fa7c6c3-8647-4e2f-828e-d088274166ae)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


