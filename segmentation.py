#!/usr/bin/env python
# coding: utf-8

# Business Need: The marketing team needs to know which customers are most likely to convert, how to target them, and what strategies can improve their spending.

# In[100]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


# In[101]:


df = pd.read_csv("Data/Mall_Customers.csv")


# In[102]:


df.head()


# ## EDA

# In[103]:


df.isna().sum()


# In[104]:


df = df.drop('CustomerID',axis=1)


# In[105]:


df.info()


# In[106]:


df.shape


# In[107]:


df.describe().T


# ## Univariate Analysis

# In[108]:


sns.histplot(df['Age'],bins=15,kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[109]:


sns.distplot(df['Annual Income (k$)'],bins=20,kde=True)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')
plt.show()


# In[110]:


sns.distplot(df['Spending Score (1-100)'],bins=20,kde=True)
plt.title('Distribution of Score')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()


# In[111]:


sns.countplot(df['Gender'])
plt.title('Count Plot of Gender')
plt.ylabel('Gender')
plt.xlabel('Frequency')
plt.show()


# ## Bivariate & multivariate Analysis

# In[112]:


# Looking athow gender, age, income, and spending score are distributed
# Visualize the relationship betweem varaibles
sns.pairplot(df, hue='Gender')
plt.show()


# In[113]:


sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=df)
plt.title('Scatter Plot of Annual Income vs Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[114]:


sns.scatterplot(x='Age', y='Spending Score (1-100)', data=df)
plt.title('Scatter Plot of Age vs Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()


# In[115]:


sns.boxplot(x='Gender', y='Spending Score (1-100)', data=df)
plt.title('Scatter Plot of Gender vs Spending Score')
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.show()


# In[116]:


sns.violinplot(x='Gender', y='Spending Score (1-100)', data=df )
plt.title('Violin Plot of Gender vs Spending Score')
plt.xlabel('Gender')
plt.ylabel('Spending Score')
plt.show()


# In[117]:


correlation_matrix = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# ## Feature Engineering

# In[118]:


# Income to spending ration
df['Income to Spending Ratio'] = df['Annual Income (k$)'] / df['Spending Score (1-100)']
# Binned age group
df['Age Group'] = pd.cut(df['Age'], bins=[18, 30, 45, 60, 80], labels=['18-30', '31-45', '46-60', '61-80'])


# ## Data Preprocessing

# In[119]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df[['Income', 'Score']] = scaled_df


# In[120]:


df.head()


# In[121]:


scaled_df


# ### Elbow Method to Determine optimal K

# In[122]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

distortions = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_df)
    distortions.append(kmeans.inertia_)

# Elbow method plot
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.show()


# ## Elbow Method

# In[123]:


# old


from sklearn.cluster import KMeans
#data = df[['Annual Income (k$)','Spending Score (1-100)']].values
inertia_values = []

for K in range(1,11):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(scaled_df)
    inertia_values.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1,11), inertia_values, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


# In[124]:


# running kmeans with selected no.of clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_df)
# Objective: Grouping customers into distinct clusters based on spending behavior and income.


# In[125]:


df.head()


# In[126]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=df, palette='Set2', s=100, alpha=0.8)
plt.title('Mall Customer Segments by Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[127]:


K = 5

# Initialize and fit K-means
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(scaled_df)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Define colors for the clusters
colors = ['r', 'g', 'b', 'c', 'm']

# Plot the clusters
plt.figure(figsize=(8, 6))

for i in range(K):  # Plot each cluster separately with a unique color
    cluster_points = scaled_df[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, color=colors[i], label=f'Cluster {i}')

# Plotting centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, linewidths=5, color='black', label='Centroids')

# Adding labels and title
plt.title('K-means Clustering with Centroids')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')

# Adding the legend
plt.legend(title="Clusters", loc='upper right')

# Show the plot
plt.show()


# * cluster 0 : mid income & mid spending
# * cluster 1 : high income & high spending
# * cluster 2 : low income & high spending
# * cluster 3 : high income & low spending
# * cluster 4 : low income & low spending

# In[128]:


df.head()


# ### Cluster profiling

# In[129]:


cluster_summary = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'median'],
    'Spending Score (1-100)': ['mean', 'median'],
    'Age': ['mean', 'median']
}).reset_index()

cluster_summary.head()


# ### Business Insights
# * Cluster 1: High-income, high-spending—likely affluent customers who may respond to exclusive offers.
# * Cluster 2: Low-income, high-spending—focus on loyalty programs to maintain their high engagement.
# * Cluster 3: High-income, low-spending—consider upselling or personalized recommendations to increase their spending.

# In[130]:


from sklearn.metrics import silhouette_score

labels = kmeans.labels_
score = silhouette_score(scaled_df,labels)
print(f"Silhouette Score : {score}")


# ## predictive Analytics

# In[136]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score



# Features for prediction
X = df[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']]  # Add other features if needed

# Target: Cluster label
y = df['Cluster']

# Scale the features
scaler = StandardScaler()
X[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']] = scaler.fit_transform(X[['Annual Income (k$)', 'Spending Score (1-100)', 'Age']])

# Save the scaler
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)



# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a RandomForestClassifier (you can also try other models like LogisticRegression, XGBoost, etc.)
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the clusters on the test data
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# #### Feature Importance Analysis

# In[137]:


importances = clf.feature_importances_
feature_names = X.columns

for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")


# In[139]:


# Save the trained model 
    
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(clf , f)


# In[ ]:




