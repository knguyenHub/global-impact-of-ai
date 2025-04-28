import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.express as pl #conda install plotly
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import r_regression
from sklearn.preprocessing import PolynomialFeatures

# 1. download conda: https://www.anaconda.com/docs/getting-started/miniconda/install#windows-command-prompt
# 2. after download create environment in VSCode terminal:
#       conda create -n DataScienceProject 
#   where DataScienceProject is the name of the environment
# 3. activate the environment:
#   conda activate DataScienceProject
# 4. install packages: 
#   conda install matplotlib scikit-learn pandas
# 5. Open your command pallet using settings gear in bottom left corner
# 6. Select Python Interpreter: Python 3.13.2 ('DataScienceProject)
# 7. Run file to make sure theres no install issues

#initalize dataset 
GlobalImpact = pd.read_csv('Global_AI_Content_Impact_Dataset.csv')
GlobalImpact.sort_values(by=['Year', 'Country', 'Industry'], ascending=True, inplace=True)
print(GlobalImpact.head())

#Data Cleaning
GlobalImpact.drop_duplicates(inplace=True)

#fill missing numeric values with the mean
mean = GlobalImpact.mean(numeric_only=True)
GlobalImpact.fillna(value=mean, inplace=True)

#Data Visualization

# #Top AI Tools
# industryGroups = GlobalImpact.groupby(['Industry', 'Top AI Tools Used']).size().unstack(fill_value=0)
# industries = industryGroups.index
# num_industries = len(industries)

# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 10))  
# axes = axes.flatten()  

# for i, industry in enumerate(industries):
#     values = industryGroups.loc[industry]
#     axes[i].pie(values, labels=values.index, textprops={'size': 'smaller'}, radius=1, startangle=90)
#     axes[i].set_title(industry)

# plt.tight_layout()
# plt.show()

# #Job Loss
# yearGroups = GlobalImpact.groupby(['Country', 'Year'])['Job Loss Due to AI (%)'].mean().unstack()
# countries= yearGroups.index
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))  
# axes = axes.flatten() 

# for i, country in enumerate(countries):
#     values = yearGroups.loc[country]
#     axes[i].bar(values.index, values.values, color='skyblue')
#     axes[i].set_title(f"Job Loss Due to AI in {country}")
#     axes[i].set_ylabel("Job Loss (%)")
#     axes[i].set_xlabel("Year")

# plt.tight_layout()
# plt.show()

# #Consumer Trust in AI
# ConsumerTrust = GlobalImpact.groupby(['Country', 'Year'])['Consumer Trust in AI (%)'].mean().unstack()
# CTcountries= ConsumerTrust.index
# fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 7))  
# axes = axes.flatten() 

# for i, country in enumerate(CTcountries):
#     values = ConsumerTrust.loc[country]
#     axes[i].bar(values.index, values.values, color='skyblue')
#     axes[i].set_title(f"Consumer Trust in AI in {country}")
#     axes[i].set_ylabel("Consumer Trust(%)")
#     axes[i].set_xlabel("Year")

# plt.tight_layout()
# plt.show()

#Data Regression

#Job loss to AI

# sns.scatterplot(data=GlobalImpact, x = 'AI Adoption Rate (%)', y = 'Job Loss Due to AI (%)', hue = 'Industry' )
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # move legend outside
# plt.tight_layout()  # adjust layout to make space
# plt.show()

#Human Collaboration rate vs Rev Inc
X = GlobalImpact[['Human-AI Collaboration Rate (%)']].values
y = GlobalImpact[['Revenue Increase Due to AI (%)']].values
sns.scatterplot(data=GlobalImpact, x= 'Human-AI Collaboration Rate (%)', y= 'Revenue Increase Due to AI (%)', hue = 'Industry')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # move legend outside
plt.tight_layout()  # adjust layout to make space

linModel = LinearRegression()
linModel.fit(X, y)
yPredicted = linModel.predict(X)

# Graph the model
plt.plot(X, yPredicted, color='blue', linewidth=2)
plt.xlabel('Human-AI Collaboration Rate (%)', fontsize=14)
plt.ylabel('Revenue Increase Due to AI (%)', fontsize=14)
plt.show()

print(
    "Revenue Increase = ",
    linModel.intercept_[0],
    " + ",
    linModel.coef_[0][0],
    "* (Human-AI Collaboration Rate (%))",
)


sns.lmplot(
    data=GlobalImpact,
    x='Human-AI Collaboration Rate (%)', y='Revenue Increase Due to AI (%)',
    hue='Industry',
    col='Industry',  # one column per industry
    col_wrap= 4,
    order=2,
    ci=None,
    height=4
)

plt.show()

#Grouping Industries
highCorrelation = ['Education', 'Manufacturing', 'Media', 'Legal']
correlationDf = GlobalImpact[GlobalImpact['Industry'].isin(highCorrelation)]
industries = correlationDf['Industry'].unique()

palette = sns.color_palette('Set2', n_colors=len(industries))
color_mapping = dict(zip(industries, palette))

# Seaborn style
sns.set(style="whitegrid")

# Create subplots
n = len(industries)

fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)

if n == 1:
    axes = [axes]

for ax, industry in zip(axes, industries):
    subset = correlationDf[correlationDf['Industry'] == industry]
    
    X = subset['Human-AI Collaboration Rate (%)'].values.reshape(-1, 1)
    y = subset['Revenue Increase Due to AI (%)'].values

    # Polynomial features and model
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    polyModel = LinearRegression()
    polyModel.fit(X_poly, y)

    # Predict smooth curve
    x_range = np.linspace(subset['Human-AI Collaboration Rate (%)'].min(), subset['Human-AI Collaboration Rate (%)'].max(), 100).reshape(-1, 1)
    x_range_poly = poly.transform(x_range)
    y_range = polyModel.predict(x_range_poly)

    # Get color for this industry
    color = color_mapping[industry]

    # Scatter plot
    sns.scatterplot(x=subset['Human-AI Collaboration Rate (%)'], y=subset['Revenue Increase Due to AI (%)'], ax=ax, color=color, s=50)
    # Polynomial curve
    ax.plot(x_range.flatten(), y_range, color=color, linewidth=2)

    # Show equation
    second_degree= polyModel.coef_[2]
    first_degree = polyModel.coef_[1]
    intercept = polyModel.coef_[0]
    equation = f"${second_degree:.2f}x^2 + {first_degree:+.2f}x + {intercept:+.2f}$"

    y_pred = polyModel.predict(X_poly)
    #Sum of Square Error
    SSError = sum((y - y_pred) ** 2)
    print("Sum of Squared Errors (SSE) of " , industry, ":", SSError)

    r = r_regression(X, np.ravel(y))[0]
    print("Correlation coefficient (r) of " , industry, ":", r)
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

    ax.set_title(industry)

plt.tight_layout()
plt.show()
