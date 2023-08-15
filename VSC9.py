import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc

#Caching is a way to prevent Streamlit app from becoming overwhelmed by re-uploading data each time a change is made to the dashboard
@st.cache_data
def get_data():
    gaming_df_encoded = pd.read_csv('data/gaming_df_encoded.csv')    
    return gaming_df_encoded

# Use get_data function to support Streamlit app
gaming_df_encoded = get_data()

st.title('Welcome to my first data science model deployment')
st.text ('In this project, I look into predicting video streaming subscriptions. I relied on data from the 2019 Deloitte Media Survey')

st.header('The dataset')
st.write(gaming_df_encoded.head())

st.subheader('52% of survey participants subscribed to a streaming service')
Subscriptions_Streaming_video_service_dist = pd.DataFrame(gaming_df_encoded['Subscriptions - Streaming video service'].value_counts())
st.bar_chart(Subscriptions_Streaming_video_service_dist)

st.header('My feature list')

st.text('Here is a list of features in my data:')
st.write(gaming_df_encoded.columns)

# Before building a logistic regression, I need to train, test, split the data
# Separate the target variable
column_to_drop = 'Subscriptions - Streaming video service'
X = gaming_df_encoded.drop(columns=[column_to_drop])
y = gaming_df_encoded[column_to_drop]

# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the number of top features to select
k = 9

# Perform SelectKBest feature selection
selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(X_train, y_train)

# Get the selected feature indices
selected_feature_indices = selector.get_support(indices=True)

# Get the selected feature names
selected_features = X_train.columns[selected_feature_indices]

# Calculate VIF for selected features
vif = pd.DataFrame()
vif["Features"] = selected_features
vif["VIF"] = [variance_inflation_factor(X_train[selected_features].values, i) for i in range(len(selected_features))]

# Remove features with VIF scores above 10
selected_features = vif[vif["VIF"] <= 10]["Features"]

# Create a table for selected features
selected_features_table = pd.DataFrame({'Features': selected_features})

# Display the selected features table
st.header('Feature selection')
st.text('The model isolated these features for this model')
st.text('Are there better variables available? Try to find a proper combination below!')
st.dataframe(selected_features_table)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data
scaler.fit(X_train[selected_features].dropna())

# Fitting / Training the Logistic Regression Model
logreg = LogisticRegression()
logreg.fit(X_train[selected_features], y_train)

# Predict on the training set
X_train_selected_scaled = scaler.transform(X_train[selected_features].dropna())
y_train_pred = logreg.predict(X_train_selected_scaled)
y_train_pred_proba = logreg.predict_proba(X_train_selected_scaled)[:, 1]

# Calculate evaluation metrics on the training set
train_accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)
recall = recall_score(y_train, y_train_pred)
f1 = f1_score(y_train, y_train_pred)
roc_auc = roc_auc_score(y_train, y_train_pred_proba)

# Add a new section to allow users to select variables and model
st.header('Model Selection')
st.text('Choose up to 5 variables and a model to see the performance')

# Allow users to select up to 5 variables
selected_variables = st.multiselect('Select up to 5 variables for the model', gaming_df_encoded.columns)

if selected_variables:
    # Prepare the data with selected features
    X_selected = X[selected_variables]

    # Split the data into training and testing sets
    X_train_sel, X_test_sel, y_train_sel, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Choose the model
    model_name = st.selectbox('Choose a model', options=['Random Forest Regression', 'Logistic Regression'])

    if model_name == 'Random Forest Regression':
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Initialize the scaler
        scaler_sel = StandardScaler()

        # Fit the scaler on the training data
        scaler_sel.fit(X_train_sel.dropna())

        # Fitting / Training the model
        model.fit(X_train_sel, y_train_sel)

        # Predict on the test set
        X_test_sel_scaled = scaler_sel.transform(X_test_sel.dropna())
        y_test_pred = model.predict(X_test_sel_scaled)     

        # Calculate MAE and MSE for the test set
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
               
        # Plot decision tree (first tree in the forest)
        st.header('Random Forest Regression - Decision Tree Visualization')
        # Get the first decision tree from the fitted Random Forest
        tree = model.estimators_[0]

        # Plot the decision tree
        plt.figure(figsize=(12, 8))
        plot_tree(tree, feature_names=X_train_sel.columns.tolist(), filled=True, rounded=True, class_names=[str(x) for x in y.unique()])
        st.pyplot(plt)

        # Export the decision tree to Graphviz format
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=X_train_sel.columns.tolist(),
            filled=True,
            rounded=True,
            special_characters=True
        )

        st.text('Metrics for the selected model:')
        st.write(f"Mean Absolute Error (MAE): {test_mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {test_mse:.2f}")
        # Calculate R2 score on the test set
        test_r2_score = r2_score(y_test, y_test_pred)
        st.write(f"R2 Score: {test_r2_score:.2f}")
        
    if model_name == 'Logistic Regression':
       
        # Initialize the scaler
        scaler_sel = StandardScaler()

        # Fit the scaler on the training data
        scaler_sel.fit(X_train_sel.dropna())

        # Fitting / Training the model
        model = LogisticRegression()
        model.fit(X_train_sel, y_train_sel)

        # Predict on the test set
        X_test_sel_scaled = scaler_sel.transform(X_test_sel.dropna())
        y_test_pred = model.predict(X_test_sel_scaled)

        # Calculate accuracy for the test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # For logistic regression, calculate y_test_pred_proba for ROC AUC
        y_test_pred_proba = model.predict_proba(X_test_sel_scaled)[:, 1]
        roc_auc_logistic = roc_auc_score(y_test, y_test_pred_proba)
        st.write(f"ROC AUC Score: {roc_auc_logistic:.2f}")
        
        # Plot ROC curve on the testing set
        plt.figure()
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # Calculate and print precision, recall, and F1 score
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)
        
        # Print evaluation metrics on the testing set
        st.write("Testing Accuracy:", test_accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)
        st.write("ROC AUC Score:", roc_auc)

        # Calculate true positives, true negatives, false positives, false negatives
        true_positives = pd.crosstab(y_test, y_test_pred)[1][1]
        true_negatives = pd.crosstab(y_test, y_test_pred)[0][0]
        false_positives = pd.crosstab(y_test, y_test_pred)[0][1]
        false_negatives = pd.crosstab(y_test, y_test_pred)[1][0]

        # Print the counts
        st.write("True Positives:", true_positives)
        st.write("True Negatives:", true_negatives)
        st.write("False Positives:", false_positives)
        st.write("False Negatives:", false_negatives)
        def mean_absolute_percentage_error(y_test, y_pred):
            return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Calculate evaluation metrics on the test set
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)  # Custom function for MAPE
