import streamlit as st
from prediction_helper import predict

# ============================================================================
# INSTALLATION & SETUP INSTRUCTIONS
# ============================================================================
# Browse to the app folder, and install xgboost in /usr/bin/python3
# >> /usr/bin/python3 -m pip install xgboost
# >> /usr/bin/python3 -m pip install scikit-learn
# To run the app in Ubuntu, type the following in terminal
# streamlit run ./main.py

# ============================================================================
# APP CONFIGURATION
# ============================================================================
# Set the title that appears at the top of the web application
st.title("Health Insurance Prediction App")

# ============================================================================
# CATEGORICAL OPTIONS DEFINITION
# ============================================================================
# Define all possible values for categorical/dropdown inputs
# These options will populate the selectbox widgets in the UI
categorical_options = {
    'Gender':            ['Male', 'Female'],
    'Marital Status':    ['Unmarried', 'Married'],
    'BMI Category':      ['Normal', 'Obesity', 'Overweight', 'Underweight'],
    'Smoking Status':    ['No Smoking', 'Regular', 'Occasional'],
    'Employment Status': ['Salaried', 'Business-Owner', 'Freelancer'],
    'Region':            ['Northwest', 'Southwest', 'Northeast', 'Southeast'],  
    'Medical History':   ['No Disease', 'Diabetes', 'High blood pressure', 'Diabetes & High blood pressure',
                          'Thyroid', 'Heart disease', 'High blood pressure & Heart disease',
                          'Diabetes & Thyroid', 'Diabetes & Heart disease'],
    'Insurance Plan':    ['Bronze','Silver','Gold']
}

# ============================================================================
# UI LAYOUT - CREATE GRID STRUCTURE
# ============================================================================
# Create a 4x3 grid layout (4 rows, 3 columns each) for organizing input widgets
# This provides a clean, organized interface for users to input their data
row1 = st.columns(3)  # First row: Age, Dependents, Income
row2 = st.columns(3)  # Second row: Genetic Risk, Insurance Plan, Employment
row3 = st.columns(3)  # Third row: Gender, Marital Status, BMI
row4 = st.columns(3)  # Fourth row: Smoking, Region, Medical History

# ============================================================================
# ROW 1 - NUMERIC INPUTS
# ============================================================================
# Position input widgets in the first row (3 columns)
with row1[0]:
    # Age input: restricted to adults (18-100 years)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)

with row1[1]:
    # Number of dependents: 0 to 10 people who rely on the insurance holder
    number_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, step=1)

with row1[2]:
    # Annual income in thousands: 0 to 200K (e.g., 50 = £50,000)
    income_thousands = st.number_input("Annual income in thousands", min_value=0, max_value=200, step=1)

# ============================================================================
# ROW 2 - MIXED INPUTS (NUMERIC + CATEGORICAL)
# ============================================================================
# Position input widgets in the second row
with row2[0]:
    # Genetic risk score: 0 (low) to 5 (high) indicating hereditary health risks
    genetical_risk = st.number_input("Genetical Risk", min_value=0, max_value=5, step=1)

with row2[1]:
    # Insurance plan tier: Bronze (basic) < Silver (medium) < Gold (premium)
    insurance_plan = st.selectbox("Insurance Plan", categorical_options['Insurance Plan'])

with row2[2]:
    # Employment type: affects income stability and insurance risk assessment
    employment_status = st.selectbox("Employment Status", categorical_options['Employment Status'])

# ============================================================================
# ROW 3 - CATEGORICAL INPUTS (DEMOGRAPHICS & HEALTH)
# ============================================================================
# Position input widgets in the third row
with row3[0]:
    # Gender: Male or Female (affects insurance risk calculations)
    gender = st.selectbox("Gender", categorical_options['Gender'])

with row3[1]:
    # Marital status: Married or Unmarried (can affect dependent coverage)
    marital_status = st.selectbox("Marital Status", categorical_options['Marital Status'])

with row3[2]:
    # BMI Category: Body Mass Index classification (health indicator)
    bmi_category = st.selectbox("BMI Category", categorical_options['BMI Category'])

# ============================================================================
# ROW 4 - CATEGORICAL INPUTS (LIFESTYLE & MEDICAL)
# ============================================================================
# Position input widgets in the fourth row
with row4[0]:
    # Smoking status: major risk factor for health insurance premiums
    smoking_status = st.selectbox("Smoking Status", categorical_options['Smoking Status'])

with row4[1]:
    # Geographic region: different regions may have different healthcare costs
    region = st.selectbox("Region", categorical_options['Region'])

with row4[2]:
    # Medical history: pre-existing conditions significantly affect premium costs
    medical_history = st.selectbox("Medical History", categorical_options['Medical History'])

# ============================================================================
# INPUT DATA COLLECTION
# ============================================================================
# Collect all user inputs into a dictionary for easy processing
# Keys match the expected format for the prediction model
input_dict = {
    "Age"                        : age,
    "Number of Dependents"       : number_of_dependents,
    "Annual income in thousands" : income_thousands,
    "Genetical Risk"             : genetical_risk,
    "Insurance Plan"             : insurance_plan,
    "Employment Status"          : employment_status,
    "Gender"                     : gender,
    "Marital Status"             : marital_status,
    "BMI Category"               : bmi_category,
    "Smoking Status"             : smoking_status,
    "Region"                     : region,
    "Medical History"            : medical_history
}

# ============================================================================
# PREDICTION TRIGGER & RESULTS DISPLAY
# ============================================================================
# Create a prediction button that triggers the ML model when clicked
if st.button("Predict"):
    # Log button click for debugging purposes
    print("Predict button clicked")
    
    # Call the predict function from prediction_helper module
    # This function will:
    #   1. Preprocess the input_dict (one-hot encoding, feature engineering)
    #   2. Apply scaling transformation
    #   3. Select appropriate model (young vs rest based on age)
    #   4. Generate prediction using trained ML model
    
    # Extract the predicted value from the array and format to 2 decimal places
    premium_value = predict(input_dict)
    
    # Display the predicted premium in a success message box (green background)
    # and format to 2 decimal places
    st.success(f"Predicted annual insurance premium is: £{premium_value:.2f}")



