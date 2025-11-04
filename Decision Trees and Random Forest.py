import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 1.First I get and print the data
loans = pd.read_csv('loan_data.csv')
print(loans.info())
print(loans.head())
print(loans.describe())

# 2. Now I do Exploratory Data Analysis
# a.Create a histogram of two FICO distributions on top of each other,
# one for each credit.policy outcome.
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# b. Same for column not.fully.paid
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# c. Create a countplot using seaborn showing the counts of loans by purpose,
# with the color hue defined by not.fully.paid.
plt.figure(figsize=(10,6))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans)
plt.show()

# d. Let's see the trend between FICO score and interest rate. Recreate the following jointplot.
sns.jointplot(x='fico',y='int.rate',data=loans)
plt.show()

# e.Create the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Check the documentation
# for lmplot() if you can't figure out how to separate it into columns.
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
plt.show()


# 3. Now I Setting up the Data
print(loans.info())

# a. Create a list of 1 element containing the string 'purpose'. Call this list cat_feats.
cat_feats = ['purpose']
#Now use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Set this dataframe as final_data.
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
print(final_data.info())

# 4. Train Test Split - Training a Decision Tree Model - Predictions and Evaluation of Decision Tree
X = final_data.drop('not.fully.paid',axis=1)
y= final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# 5. Training the Random Forest model
rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))