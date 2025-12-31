import pandas as pd
from joblib import load

# ============================================================================
# MODEL AND SCALER LOADING
# ============================================================================
# Load pre-trained machine learning models from the artifacts folder
# Two separate models are used based on age groups for better accuracy:
# - model_young: trained specifically for ages <= 25
# - model_rest: trained for ages > 25
model_rest  = load("artifacts/model_rest.joblib")
model_young = load("artifacts/model_young.joblib")

# Load pre-fitted scaler objects (MinMaxScaler) from the artifacts folder
# These scalers normalize input features to 0-1 range matching training data
# Separate scalers are used for each age group to match their respective models
scaler_rest  = load("artifacts/scaler_rest.joblib")
scaler_young = load("artifacts/scaler_young.joblib")

# ============================================================================
# FEATURE COLUMN DEFINITIONS
# ============================================================================
# Define the exact column names expected by the model (18 features after VIF analysis)
# These columns match the training data structure after one-hot encoding
# Note: 'income_level_encode' was dropped during VIF analysis and is NOT included here
expected_columns = ['age', 'number_of_dependants', 'income_thousands', 'genetical_risk', 'total_risk_scores',
                    'insurance_plan_encode', 'gender_male', 'marital_status_unmarried',
                    'region_northwest', 'region_southeast', 'region_southwest',
                    'bmi_category_obesity', 'bmi_category_overweight', 'bmi_category_underweight',
                    'smoking_status_occasional', 'smoking_status_regular',
                    'employment_status_freelancer', 'employment_status_salaried'
                    ]

# ============================================================================
# ENCODING DICTIONARIES
# ============================================================================
# Ordinal encoding for insurance plans (Bronze < Silver < Gold)
# Higher number = more comprehensive coverage
insurance_plan_encoding = {'Bronze': 1,
                           'Silver': 2,
                           'Gold': 3}

# NOTE: income_level_encode is commented out because it was dropped after VIF analysis
# VIF (Variance Inflation Factor) analysis showed high multicollinearity with 'income_thousands'
# The column was removed to prevent redundancy and improve model performance
'''
income_level_encode = {'<10K'     :1,
                       '10K - 25K':2,
                       '25K - 40K':3,
                       '> 40K'    :4 }
'''

# ============================================================================
# MAIN PREPROCESSING FUNCTION
# ============================================================================
def preprocess_input(input_dict):
    """
    Convert raw user input into model-ready format.
    
    Process:
    1. Create DataFrame with expected columns initialized to 0
    2. Parse input_dict and set appropriate column values
    3. Apply one-hot encoding for categorical variables
    4. Calculate total risk scores from medical history
    5. Apply scaling transformation
    
    Args:
        input_dict: Dictionary containing user inputs from Streamlit UI
        
    Returns:
        scaled_df: DataFrame with 18 features, scaled and ready for prediction
    """
    
    # Define risk scores for different medical conditions
    # Business logic: more severe conditions get higher scores
    risk_scores_encoding = {"none": 0,
                            "no disease": 0,
                            "thyroid": 5,
                            "diabetes": 6,
                            "high blood pressure": 6,
                            "heart disease": 8}

    # Initialize DataFrame with all expected columns set to 0 (default for one-hot encoding)
    # This ensures all binary columns exist even if user doesn't select those options
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    # ========================================================================
    # PARSE INPUT DICTIONARY AND POPULATE DATAFRAME
    # ========================================================================
    # Loop through all user inputs and set corresponding column values
    # Uses one-hot encoding: set column to 1 if condition matches, else remains 0
    
    for key, value in input_dict.items():
        
        # Gender encoding (one-hot): only 'Male' sets gender_male=1, Female remains 0
        if key == 'Gender' and value == 'Male':
            df['gender_male'] = 1

        # Region encoding (one-hot): set appropriate region column to 1
        if key == 'Region' and value == 'Northwest':
            df['region_northwest'] = 1
        elif key == 'Region' and value == 'Southwest':
            df['region_southwest'] = 1  
        elif key == 'Region' and value == 'Southeast':
            df['region_southeast'] = 1

        # Marital Status encoding (one-hot): only 'Unmarried' sets column to 1
        if key == 'Marital Status' and value == 'Unmarried':
            df['marital_status_unmarried'] = 1

        # BMI Category encoding (one-hot): set appropriate BMI column to 1
        # Normal BMI remains all 0s (reference category)
        if key == 'BMI Category' and value == 'Underweight':
            df['bmi_category_underweight'] = 1
        elif key == 'BMI Category' and value == 'Overweight':
            df['bmi_category_overweight'] = 1
        elif key == 'BMI Category' and value == 'Obesity':
            df['bmi_category_obesity'] = 1

        # Smoking Status encoding (one-hot): set appropriate smoking column to 1
        # 'No Smoking' remains all 0s (reference category)
        if key == 'Smoking Status' and value == 'Occasional':
            df['smoking_status_occasional'] = 1
        elif key == 'Smoking Status' and value == 'Regular':
            df['smoking_status_regular'] = 1

        # Employment Status encoding (one-hot): set appropriate employment column to 1
        # 'Business-Owner' remains all 0s (reference category)
        if key == 'Employment Status' and value == 'Freelancer':
            df['employment_status_freelancer'] = 1
        elif key == 'Employment Status' and value == 'Salaried':
            df['employment_status_salaried'] = 1

        # Insurance Plan encoding (ordinal): Bronze=1, Silver=2, Gold=3
        # Uses .get() with default=1 (Bronze) if value not found
        if key == 'Insurance Plan':
            df['insurance_plan_encode'] = insurance_plan_encoding.get(value, 1)

        # Numeric features: directly assign values to DataFrame
        if key == 'Age':
            df['age'] = value

        if key == 'Number of Dependents':
            df['number_of_dependants'] = value

        if key == 'Annual income in thousands':
            df['income_thousands'] = value

        if key == 'Genetical Risk':
            df['genetical_risk'] = value

        # Medical History: calculate total risk score from disease combinations
        # E.g., "Diabetes & Heart disease" = 6 + 8 = 14
        if key == 'Medical History':
            df['total_risk_scores'] = calculate_total_risk_scores(input_dict['Medical History'], risk_scores_encoding)

    # Apply scaling transformation based on age group
    df = handle_scaler(input_dict['Age'], df)
    
    return df


# ============================================================================
# SCALING HANDLER FUNCTION
# ============================================================================
def handle_scaler(age, df):
    """
    Apply appropriate scaler transformation based on user's age.
    
    Important: The scaler was fitted on 19 features (including income_level_encode),
    but the model expects 18 features (without income_level_encode).
    This function handles that discrepancy.
    
    Args:
        age: User's age (determines which scaler/model to use)
        df: DataFrame with 18 features
        
    Returns:
        scaled_df: Scaled DataFrame with 18 features ready for model prediction
    """
    
    # Select appropriate scaler based on age threshold
    # Age <= 25: use young scaler (trained on younger population data)
    # Age > 25: use rest scaler (trained on general population data)
    if age <= 25:
        scaler_obj = scaler_young
    else:
        scaler_obj = scaler_rest

    # CRITICAL FIX: Add temporary column to match scaler's expected 19 features
    # During training, scaler was fitted BEFORE VIF analysis dropped income_level_encode
    # We add it temporarily with value 0, transform, then drop it again
    df['income_level_encode'] = 0
    
    # Debug output: verify column counts
    print(f"\n df.columns({len(df.columns)}) : , {df.columns}")
    print(f"\n scaler_obj.feature_names_in_({len(scaler_obj.feature_names_in_)}) : , {scaler_obj.feature_names_in_}")

    # Apply MinMaxScaler transformation (scales all features to 0-1 range)
    # Must use scaler's expected column order for correct transformation
    cols_to_org_scale = scaler_obj.transform(df[scaler_obj.feature_names_in_])

    # Convert scaled numpy array back to DataFrame for easier manipulation
    scaled_df = pd.DataFrame(
        cols_to_org_scale,
        columns=scaler_obj.feature_names_in_
    )

    # Drop the temporary income_level_encode column to match model's expected 18 features
    # This column was removed during VIF analysis due to high correlation with income_thousands
    scaled_df = scaled_df.drop('income_level_encode', axis=1)

    # Debug output: verify final shape is (1, 18)
    print(f"\n After dropping, scaled_df.shape: {scaled_df.shape}")

    return scaled_df


# ============================================================================
# RISK SCORE CALCULATION FUNCTION
# ============================================================================
def calculate_total_risk_scores(medical_history, risk_scores_encoding):
    """
    Calculate total risk score from medical history string.
    
    Handles multiple diseases separated by ' & ' and sums their risk scores.
    
    Examples:
        - "No Disease" → 0
        - "Diabetes" → 6
        - "Diabetes & Heart disease" → 6 + 8 = 14
        - "High blood pressure & Heart disease" → 6 + 8 = 14
    
    Args:
        medical_history: String describing medical conditions (from user input)
        risk_scores_encoding: Dictionary mapping disease names to risk scores
        
    Returns:
        total_risk_score: Sum of all disease risk scores
    """
    
    # Split medical history by ' & ' to handle multiple diseases
    # Convert to lowercase for case-insensitive matching
    diseases = medical_history.lower().split(" & ")
    
    # Sum risk scores for all diseases, default to 0 if disease not in dictionary
    total_risk_score = sum(risk_scores_encoding.get(disease, 0) for disease in diseases)
    
    return total_risk_score


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================
def predict(input_dict):
    """
    Generate insurance premium prediction from user inputs.
    
    Process:
    1. Preprocess input (encoding, feature engineering, scaling)
    2. Select appropriate model based on age
    3. Generate prediction
    4. Return as integer (premium in pounds)
    
    Args:
        input_dict: Dictionary containing all user inputs from Streamlit UI
        
    Returns:
        prediction: Predicted annual insurance premium (integer, in pounds)
    """
    
    # Debug output: print raw input dictionary
    print(input_dict)
    
    # Preprocess input into model-ready format (18 scaled features)
    input_df = preprocess_input(input_dict)

    # Select appropriate model based on age threshold
    # Younger people (<=25) have different risk patterns, so use specialized model
    if input_dict['Age'] <= 25:
        model = model_young
    else:
        model = model_rest

    # Generate prediction using selected model
    # Model returns numpy array, e.g., [12345.67]
    prediction = model.predict(input_df)

    # Convert prediction to integer (round to nearest pound)
    # Extract first element from array and cast to int
    return int(prediction)
