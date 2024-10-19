import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the scaler and Random Forest model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:  # Change to your Random Forest model file
    rf_model = pickle.load(f)

st.title("Mall Customer Segmentation")
st.write("This app predicts which customer segment a customer belongs to based on their age, annual income, and spending score.")    

# Define the user input function
def user_input():
    
    annual_income = st.sidebar.number_input("Annual Income (k$)", min_value=0, max_value=150, value=50)
    spending_score = st.sidebar.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    age = st.sidebar.slider("Age", min_value=18, max_value=80, value=30)



    # Create a DataFrame with the input features
    data = {
        'Annual Income (k$)': annual_income,
        'Spending Score (1-100)': spending_score,
        'Age': age
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input()

# Display the input DataFrame for debugging
st.write("Customer Data Input:")
st.write(input_df)

# Scale the input features for prediction
try:
    # Scale the input features
    scaled_input = scaler.transform(input_df)

    # Predict the cluster using the Random Forest model
    prediction = rf_model.predict(scaled_input)

    # Show prediction results
    st.write(f"The predicted cluster is: {prediction[0]}")
except ValueError as e:
    st.error(f"Error during prediction: {e}")


# Show marketing insights for each cluster
def show_insights(cluster):
    insights = {
        0: ("Mid Income & Mid Spending", "🟢", "Promotions: Limited-time discounts and value bundles.  Loyalty Programs:  Rewards to encourage repeat purchases."),
        1: ("High Income & High Spending", "🟣", "Exclusive Offers: Access to new products/events.  Premium Products:  Introduce luxury items."),
        2: ("Low Income & High Spending", "🔵", "Affordable Luxury: Position products as treats.   Flexible Payment:  Offer installment plans."),
        3: ("High Income & Low Spending", "🟡", "Personalized Recommendations: Tailored suggestions.  Engagement Campaigns:  Educate on product benefits."),
        4: ("Low Income & Low Spending", "🔴", "Value-Focused Campaigns: Promote budget-friendly options.  Community Engagement:  Build local loyalty.")
    }
    
    if cluster in insights:
        title, emoji, details = insights[cluster]
        st.subheader(f"{emoji} {title}")
        st.write(details)

# After predicting the cluster, display insights
show_insights(prediction[0])



