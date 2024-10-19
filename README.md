# Mall Customer Segmentation

This project is a customer segmentation analysis using **K-Means Clustering** and **Random Forest**. The goal is to help the marketing team understand customer behavior and target customers more effectively.

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [Cluster Insights](#cluster-insights)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [How to Run the App](#how-to-run-the-app)
- [Contributing](#contributing)

## Project Overview

This project aims to segment mall customers into distinct groups based on their **Annual Income**, **Spending Score**, **Age**, and **Gender**. Using this segmentation, we provide business insights to help the marketing team plan targeted strategies and improve conversion rates.

## Business Problem

The marketing team needs to know:
- Which customers are most likely to convert?
- How to effectively target them with personalized marketing strategies?
- What actions can be taken to improve spending behavior?

## Dataset

The dataset includes the following features:
- **Customer ID**: Unique identifier for each customer.
- **Age**: Customer age.
- **Gender**: Customer gender.
- **Annual Income (k$)**: Customer's annual income.
- **Spending Score (1-100)**: A score assigned based on customer behavior and spending patterns.

Dataset source: [Kaggle Mall Customers Dataset](https://www.kaggle.com/datasets)

## Solution Approach

1. **Data Preprocessing**:
   - Feature scaling using **StandardScaler**.
   - Encoding **Gender** as binary values.
   
2. **Modeling**:
   - **K-Means Clustering** for segmenting customers into 5 clusters.
   - **Random Forest** classifier to predict customer clusters for new data.

## Cluster Insights

- **Cluster 0: Mid Income & Mid Spending**  
  Moderate potential customers, good for loyalty programs.

- **Cluster 1: High Income & High Spending**  
  High-value customers; focus on premium services.

- **Cluster 2: Low Income & High Spending**  
  Value seekers; good for discount promotions.

- **Cluster 3: High Income & Low Spending**  
  Upsell opportunities exist; consider personalized offers.

- **Cluster 4: Low Income & Low Spending**  
  Limited immediate potential, but could be targeted for growth.

## Technologies Used

- **Python**
- **Streamlit**: For interactive web app development.
- **Scikit-learn**: For machine learning models.
- **Pandas**: Data manipulation.
- **Matplotlib & Seaborn**: Visualization.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Mall-Customer-Segmentation.git

2. Navigate to the project directory:
   ```bash
   cd Mall-Customer-Segmentation
   
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   
## How to run the app

1. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
2. Open your browser to http://localhost:8501 to interact with the customer segmentation dashboard.

## Contributing 
Feel free to submit pull requests or open issues if you would like to contribute to the project.

You can copy and paste this directly into your `README.md` file! Let me know if you need anything else.

   
