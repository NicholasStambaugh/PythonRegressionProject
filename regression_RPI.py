import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Program Files\JetBrains\RE.csv")


# print the columns present in the dataset
print(df.columns)

# print the top 5 rows in the dataset
print(df.head())

# print number of missing values
df.isna().sum()

# plotting heatmap, find correlating values
sns.heatmap(df.corr(), square=True, cmap="RdYlGn")

#Simple Linear Regression
sns.lmplot(x="num_conv", y="price", data = df)

# Initializing the variables X and Y
X = df[["num_conv"]]
y = df[["price"]]

# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

# Fitting the training data to our model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# check prediction score
regressor.score(X_test, y_test)

# predict the y values
y_pred=regressor.predict(X_test)

# a data frame with actual and predicted values of y
evaluate = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
evaluate.head(10)
#barplot of differences between actual and predicted y values
evaluate.head(10).plot(kind ="bar")

## MULTIPLE LINEAR REGRESSION ##

# Preparing the data
X = df[["num_conv", "lat", "long", "dist_hosp"]]
y = df[["price"]]

# Splitting the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

# Fitting the training data to our model
regressor.fit(X_train, y_train)

#score of this model
regressor.score(X_test, y_test)

# predict the y values
y_pred=regressor.predict(X_test)

# a data frame with actual and predicted values of y
evaluate = pd.DataFrame({"Actual": y_test.values.flatten(), "Predicted": y_pred.flatten()})
evaluate.head(10)

evaluate.head(10).plot(kind="bar")

#more visuals, not working currently...
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
cf.go_offline()

df.iplot(kind='scatter', x='num_conv', y='price', mode='markers', color = '#5d3087',  layout = {
        'title':'Size vs Price',
        'xaxis': {'title': 'Size', 'type': 'log'},
        'yaxis': {'title': "Price"}
    })
#show plots
plt.show()
