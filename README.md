# Credit-Card-Fraud-Detection
import timeit
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import streamlit as st

warnings.filterwarnings("ignore")

# Streamlit title
st.title('Credit Card Fraud Detection!')

@st.cache_data
def load_data():
    return pd.read_csv('creditcard.csv')

# Load data
df = load_data()
if df is None or df.empty:
    st.error('Data could not be loaded')

# Show initial data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('Shape of the dataframe: ', df.shape)
    st.write('Data description: \n', df.describe())

# Fraud and valid transaction details
fraud = df[df.Class == 1]
valid = df[df.Class == 0]
outlier_percentage = (len(fraud) / len(valid)) * 100

if st.sidebar.checkbox('Show fraud and valid transaction details'):
    st.write('Fraudulent transactions are: %.3f%%' % outlier_percentage)
    st.write('Fraud Cases: ', len(fraud))
    st.write('Valid Cases: ', len(valid))

# Preparing features and labels
X = df.drop(['Class'], axis=1)
y = df.Class

# Split the data into training and testing sets
size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=42)

# Show shapes of training and test sets
if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
    st.write('X_train: ', X_train.shape)
    st.write('y_train: ', y_train.shape)
    st.write('X_test: ', X_test.shape)
    st.write('y_test: ', y_test.shape)

# Import models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'k Nearest Neighbor': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Extra Trees': ExtraTreesClassifier(random_state=42)
}

# Feature importance function
def feature_sort(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model.feature_importances_ if hasattr(model, 'feature_importances_') else np.zeros(X_train.shape[1])

# Classifiers for feature importance
mod_feature = st.sidebar.selectbox('Which model for feature importance?', ['Extra Trees', 'Random Forest'])

# Measure execution time
start_time = timeit.default_timer()
importance = feature_sort(models[mod_feature], X_train, y_train)
elapsed = timeit.default_timer() - start_time
st.write('Execution Time for feature selection: %.2f seconds' % elapsed)

# Plot feature importance
if st.sidebar.checkbox('Show plot of feature importance'):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importance)), importance)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Feature (Variable Number)')
    ax.set_ylabel('Importance')
    st.pyplot(fig)
    plt.close(fig)

# Select top features
n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)
# Get indices of top features in descending order of importance
top_features_indices = np.argsort(importance)[-n_top_features:][::-1]
top_features = X_train.columns[top_features_indices]

if st.sidebar.checkbox('Show selected top features'):
    st.write(f'Top {n_top_features} features in order of importance are: {top_features.tolist()}')
    
# Scale selected features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.loc[:, top_features])
X_test_scaled = scaler.transform(X_test.loc[:, top_features])

# Performance computation
def compute_performance(model, X_train, y_train, X_test, y_test):
    start_time = timeit.default_timer()
    scores = cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy').mean()
    st.write(f"Cross-Validation Accuracy: {scores:.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # Format classification report as DataFrame
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.round(4)  # Round values for better readability
    
    # Highlight important metrics for better visibility
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    styled_report_df = report_df.style.apply(highlight_max, subset=['precision', 'recall', 'f1-score'])
    st.write("Classification Report:")
    st.dataframe(styled_report_df)  # Display as DataFrame for better formatting

    mcc = matthews_corrcoef(y_test, y_pred)
    st.write(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

    elapsed = timeit.default_timer() - start_time
    st.write(f"Execution Time for performance computation: {elapsed:.2f} seconds")

# Handle imbalanced data
imbalance_methods = {
    'No Rectifier': lambda X, y: (X, y),
    'SMOTE': lambda X, y: (SMOTE(random_state=42).fit_resample(X, y)),
    'Near Miss': lambda X, y: (NearMiss().fit_resample(X, y))
}

if st.sidebar.checkbox('Run a credit card fraud detection model'):
    classifier = st.sidebar.selectbox('Which algorithm?', list(models.keys()))
    imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', list(imbalance_methods.keys()))

    X_train_bal, y_train_bal = imbalance_methods[imb_rect](X_train_scaled, y_train)
    st.write('Shape of balanced y_train: ', np.bincount(y_train_bal))
    
    compute_performance(models[classifier], X_train_bal, y_train_bal, X_test_scaled, y_test)

