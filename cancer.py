import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import plotly.express as px

# Step 1: Load the data
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\mk994\Downloads\data (1).csv")  # Adjust the path to your dataset
    return data

data = load_data()

# Step 2: Preprocess the data
def preprocess_data(data):
    # Drop unnecessary columns
    data = data.drop(columns=["Unnamed: 32", "id"])

    # Check for missing values in the dataset
    if data.isnull().sum().any():
        st.warning("The dataset contains missing values. These will be filled with column means.")
        
        # Fill missing values with the mean of each column
        data.fillna(data.mean(), inplace=True)
    
    # Encode the 'diagnosis' column (M=1, B=0)
    labelencoder = LabelEncoder()
    data['diagnosis'] = labelencoder.fit_transform(data['diagnosis'])
    
    # Split the data into features (X) and target (y)
    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]
    
    # Normalize the feature data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, scaler

X, y, scaler = preprocess_data(data)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a simple deep learning model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(X_train.shape[1])

# Step 5: Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Streamlit application interface
st.title("Breast Cancer Detection")

# Step 7: Display unique Patient IDs for selection
unique_patient_ids = data.index.unique()  # Get unique patient indices

# Use st.selectbox to allow selection of a single ID with scrolling
selected_id = st.selectbox(
    "Select Patient ID:",
    options=unique_patient_ids,
    index=0  # Default to the first ID
)

# Prediction for the selected patient ID
if st.button("Detect Cancer"):
    # Extract patient features
    patient_data = data.drop(columns=['diagnosis', 'Unnamed: 32', 'id']).iloc[selected_id]

    # Ensure the sample data has the same shape
    patient_features = patient_data.values.reshape(1, -1)

    # Impute missing values if necessary
    if np.isnan(patient_features).any():
        patient_features = np.nan_to_num(patient_features, nan=np.nanmean(patient_features))

    # Standardize the features using the same scaler
    patient_features = scaler.transform(patient_features)

    # Make a prediction
    prediction = model.predict(patient_features)
    prediction_label = (prediction > 0.5).astype(int)

    # Display results for the patient
    st.subheader(f"Patient ID: {selected_id}")
    if prediction_label == 1:
        st.error("The patient is likely to have breast cancer (Malignant).")
        st.subheader("Recommended Tips:")
        st.write(""" 
            1. Consult with a medical professional immediately for a detailed diagnosis.
            2. Consider treatment options such as surgery, chemotherapy, or radiation.
            3. Maintain a healthy diet and lifestyle to support treatment.
            4. Regular follow-up and screenings.
            5. Stay positive and seek support from friends, family, or counseling.
        """)
    else:
        st.success("The patient is likely to have a benign condition (Non-cancerous).")
        st.subheader("Prevention Tips for the Future:")
        st.write(""" 
            1. Maintain a healthy weight.
            2. Stay physically active and engage in regular exercise.
            3. Eat a healthy diet rich in fruits and vegetables.
            4. Avoid smoking and limit alcohol consumption.
            5. Regularly screen for breast cancer, especially if you have a family history.
        """)

    # Plot 1: Feature Distribution Plot
    features = data.drop(columns=['diagnosis', 'Unnamed: 32', 'id'])
    feature_means = features.mean()  # Get mean of features for comparison
    feature_data = pd.DataFrame({
        'Feature': features.columns,
        'Patient Value': patient_data.values.flatten(),
        'Mean Value': feature_means.values
    })

    fig1 = px.bar(feature_data, x='Feature', y=['Patient Value', 'Mean Value'],
                  title="Feature Distribution: Patient vs Average",
                  labels={'value': 'Value', 'Feature': 'Feature'},
                  barmode='group')
    st.plotly_chart(fig1)

    # Plot 2: Prediction Probability Plot
    y_train_prob = model.predict(X_train)  # Get training probabilities
    fig2 = px.histogram(y_train_prob, nbins=30, title='Distribution of Predicted Probabilities',
                         labels={'value': 'Predicted Probability'},
                         marginal='rug')
    st.plotly_chart(fig2)
