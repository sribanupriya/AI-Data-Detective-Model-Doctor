
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="AI Data Detective & Model Doctor",
    page_icon="",
    layout="wide"
)

st.title(" AI Data Detective & Model Doctor")

st.write(
    "Automated Data Quality Analysis and Machine Learning "
    "Model Diagnosis System"
)


# DATA DETECTIVE


st.header(" Data Detective")

uploaded_file = st.file_uploader(
    " Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(" Dataset uploaded successfully!")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            int(df.duplicated().sum())
        )

    st.subheader(" Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    # Missing values
    st.subheader(" Missing Value Analysis")

    missing_values = df.isnull().sum()

    missing_report = pd.DataFrame({
        "Column": missing_values.index,
        "Missing Values": missing_values.values
    })

    st.dataframe(
        missing_report,
        use_container_width=True
    )

    # Data types
    st.subheader("Data Type Analysis")

    datatype_report = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        datatype_report,
        use_container_width=True
    )


# MODEL DOCTOR


st.header(" Model Doctor")

st.write(
    "Enter new student information to generate "
    "a machine learning prediction."
)

age = st.number_input(
    "Age",
    min_value=15,
    max_value=100,
    value=21
)

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=90.0
)

previous_marks = st.number_input(
    "Previous Marks (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

if st.button(" Predict Placement"):

    try:

        # Load saved model and scaler
        model = joblib.load(
            "logistic_regression_model.pkl"
        )

        scaler = joblib.load(
            "feature_scaler.pkl"
        )

        # Create input dataframe
        new_student = pd.DataFrame({
            "Age": [age],
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks]
        })

        # Scale input
        new_student_scaled = scaler.transform(
            new_student
        )

        # Prediction
        prediction = model.predict(
            new_student_scaled
        )

        probability = model.predict_proba(
            new_student_scaled
        )

        if prediction[0] == 1:

            result = "Yes"
            confidence = probability[0][1]

        else:

            result = "No"
            confidence = probability[0][0]

        st.subheader("Prediction Result")

        if result == "Yes":
            st.success(
                " Predicted Placement: YES"
            )
        else:
            st.warning(
                "Predicted Placement: NO"
            )

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

        st.metric(
            "Placement Probability",
            f"{probability[0][1] * 100:.2f}%"
        )

        st.success(
            " Prediction completed successfully!"
        )

    except FileNotFoundError:

        st.error(
            " Saved model files were not found. "
            "Please keep the .pkl files in the same folder as app.py."
        )




# MODEL COMPARISON


st.header(" Model Comparison")

st.write(
    "Compare Logistic Regression and Decision Tree "
    "predictions for the new student."
)

if st.button(" Compare ML Models"):

    try:

        logistic_model = joblib.load(
            "logistic_regression_model.pkl"
        )

        decision_tree_model = joblib.load(
            "decision_tree_model.pkl"
        )

        scaler = joblib.load(
            "feature_scaler.pkl"
        )

        comparison_input = pd.DataFrame({
            "Age": [age],
            "Study_Hours": [study_hours],
            "Attendance": [attendance],
            "Previous_Marks": [previous_marks]
        })

        # Logistic Regression
        scaled_input = scaler.transform(
            comparison_input
        )

        lr_prediction = logistic_model.predict(
            scaled_input
        )

        lr_probability = logistic_model.predict_proba(
            scaled_input
        )

        # Decision Tree
        dt_prediction = decision_tree_model.predict(
            comparison_input
        )

        dt_probability = decision_tree_model.predict_proba(
            comparison_input
        )

        lr_result = (
            "Yes" if lr_prediction[0] == 1
            else "No"
        )

        dt_result = (
            "Yes" if dt_prediction[0] == 1
            else "No"
        )

        st.subheader(" Prediction Comparison")

        col1, col2 = st.columns(2)

        with col1:
            st.info(" Logistic Regression")
            st.write("Prediction:", lr_result)
            st.write(
                f"Placement Probability: "
                f"{lr_probability[0][1] * 100:.2f}%"
            )

        with col2:
            st.info(" Decision Tree")
            st.write("Prediction:", dt_result)
            st.write(
                f"Placement Probability: "
                f"{dt_probability[0][1] * 100:.2f}%"
            )

        comparison_data = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Decision Tree"
            ],
            "Prediction": [
                lr_result,
                dt_result
            ],
            "Placement Probability (%)": [
                lr_probability[0][1] * 100,
                dt_probability[0][1] * 100
            ]
        })

        st.subheader(" Comparison Table")

        st.dataframe(
            comparison_data.round(2),
            use_container_width=True
        )

        st.success(
            " Model comparison completed!"
        )

    except FileNotFoundError:
        st.error(
            " Model files not found. "
            "Keep all .pkl files in the same folder as app.py."
        )



# ADVANCED DATA QUALITY ANALYSIS


st.header(" Advanced Data Quality Analysis")

if uploaded_file is not None:

    st.subheader(" Data Quality Score")

    # Missing values
    total_missing = int(
        df.isnull().sum().sum()
    )

    # Duplicate rows
    total_duplicates = int(
        df.duplicated().sum()
    )

    # Numeric columns
    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    # Count outliers
    total_outliers = 0

    for column in numeric_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_limit) |
            (df[column] > upper_limit)
        ]

        total_outliers += len(outliers)

    # Quality score
    score = 100

    if total_missing > 0:
        score -= 20

    if total_duplicates > 0:
        score -= 20

    if total_outliers > 0:
        score -= 10

    score = max(score, 0)

    st.metric(
        " Data Quality Score",
        f"{score}/100"
    )

    # Quality status
    if score >= 80:
        st.success(
            " Good Data Quality"
        )

    elif score >= 60:
        st.warning(
            " Moderate Data Quality"
        )

    else:
        st.error(
            "Poor Data Quality"
        )

    # Quality summary
    quality_summary = pd.DataFrame({
        "Issue": [
            "Missing Values",
            "Duplicate Rows",
            "Outliers"
        ],
        "Count": [
            total_missing,
            total_duplicates,
            total_outliers
        ]
    })

    st.subheader("Data Quality Summary")

    st.dataframe(
        quality_summary,
        use_container_width=True
    )

    # Outlier details
    st.subheader(" Outlier Detection")

    if len(numeric_columns) == 0:

        st.info(
            "No numerical columns found."
        )

    else:

        for column in numeric_columns:

            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_limit = Q1 - 1.5 * IQR
            upper_limit = Q3 + 1.5 * IQR

            outliers = df[
                (df[column] < lower_limit) |
                (df[column] > upper_limit)
            ]

            if len(outliers) > 0:

                st.warning(
                    f" {column}: "
                    f"{len(outliers)} outlier(s) detected"
                )

                st.dataframe(
                    outliers,
                    use_container_width=True
                )

            else:

                st.success(
                    f" {column}: No outliers detected"
                )

    st.success(
        " Advanced Data Quality Analysis Completed!"
    )


st.header("Automatic Data Cleaning")

if uploaded_file is not None:

    st.write(
        "Automatically clean missing values and duplicate rows."
    )

    if st.button("Clean Dataset Automatically"):

        cleaned_df = df.copy()

        duplicates_before = int(
            cleaned_df.duplicated().sum()
        )

        cleaned_df = cleaned_df.drop_duplicates()

        numeric_columns = cleaned_df.select_dtypes(
            include=np.number
        ).columns

        numeric_filled = 0

        for column in numeric_columns:

            missing_count = int(
                cleaned_df[column].isnull().sum()
            )

            if missing_count > 0:

                median_value = cleaned_df[column].median()

                cleaned_df[column] = cleaned_df[column].fillna(
                    median_value
                )

                numeric_filled += missing_count

        categorical_columns = cleaned_df.select_dtypes(
            exclude=np.number
        ).columns

        categorical_filled = 0

        for column in categorical_columns:

            missing_count = int(
                cleaned_df[column].isnull().sum()
            )

            if missing_count > 0:

                mode_values = cleaned_df[column].mode()

                if not mode_values.empty:

                    mode_value = mode_values.iloc[0]

                    cleaned_df[column] = cleaned_df[column].fillna(
                        mode_value
                    )

                    categorical_filled += missing_count

        st.subheader("Cleaning Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Duplicate Rows Removed",
                duplicates_before
            )

        with col2:
            st.metric(
                "Numeric Values Filled",
                numeric_filled
            )

        with col3:
            st.metric(
                "Categorical Values Filled",
                categorical_filled
            )

        remaining_missing = int(
            cleaned_df.isnull().sum().sum()
        )

        st.metric(
            "Remaining Missing Values",
            remaining_missing
        )

        if remaining_missing == 0:
            st.success(
                "Dataset cleaned successfully!"
            )
        else:
            st.warning(
                "Some missing values still remain."
            )

        st.subheader("Cleaned Dataset")

        st.dataframe(
            cleaned_df,
            use_container_width=True
        )

        cleaned_csv = cleaned_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Cleaned CSV",
            data=cleaned_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

        st.success(
            "Cleaned CSV is ready for download!"
        )


st.header("Data Visualization")

if uploaded_file is not None:

    st.subheader("Numerical Data Distribution")

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) > 0:

        selected_column = st.selectbox(
            "Select a numerical column",
            numeric_columns
        )

        st.line_chart(
            df[selected_column]
        )

    else:

        st.info(
            "No numerical columns found."
        )

    st.subheader("Category Distribution")

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if len(categorical_columns) > 0:

        selected_category = st.selectbox(
            "Select a categorical column",
            categorical_columns
        )

        category_counts = (
            df[selected_category]
            .value_counts()
        )

        st.bar_chart(
            category_counts
        )

    else:

        st.info(
            "No categorical columns found."
        )

    st.success(
        "Data visualization completed successfully!"
    )


st.header("Correlation Analysis")

if uploaded_file is not None:

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        correlation_matrix = numeric_df.corr()

        st.subheader("Correlation Matrix")

        st.dataframe(
            correlation_matrix.round(2),
            use_container_width=True
        )

        st.subheader("Correlation Heatmap")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        image = ax.imshow(
            correlation_matrix,
            cmap="coolwarm"
        )

        fig.colorbar(
            image,
            ax=ax
        )

        ax.set_xticks(
            range(len(correlation_matrix.columns))
        )

        ax.set_yticks(
            range(len(correlation_matrix.columns))
        )

        ax.set_xticklabels(
            correlation_matrix.columns,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            correlation_matrix.columns
        )

        ax.set_title(
            "Feature Correlation Heatmap"
        )

        st.pyplot(fig)

        st.success(
            "Correlation analysis completed successfully!"
        )

    else:

        st.info(
            "At least two numerical columns are required "
            "for correlation analysis."
        )


st.header("Automatic ML Model Training")

if uploaded_file is not None:

    st.write(
        "Select a target column and train a machine learning model automatically."
    )

    target_column = st.selectbox(
        "Select Target Column",
        df.columns
    )

    if st.button("Train Machine Learning Model"):

        try:

            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                target_encoder = LabelEncoder()

                y = target_encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            model = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy_value = accuracy_score(
                y_test,
                predictions
            )

            st.subheader("Model Training Result")

            st.success(
                "Machine learning model trained successfully!"
            )

            st.metric(
                "Test Accuracy",
                f"{accuracy_value * 100:.2f}%"
            )

            st.write(
                "Training Rows:",
                len(X_train)
            )

            st.write(
                "Testing Rows:",
                len(X_test)
            )

        except Exception as error:

            st.error(
                f"Model training error: {error}"
            )


st.header("Model Evaluation")

if uploaded_file is not None:

    st.write(
        "Evaluate the trained machine learning model."
    )

    target_column = st.selectbox(
        "Select Target Column for Evaluation",
        df.columns,
        key="evaluation_target"
    )

    if st.button("Evaluate Model"):

        try:

            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                confusion_matrix,
                ConfusionMatrixDisplay
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                encoder = LabelEncoder()

                y = encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            model = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy_value = accuracy_score(
                y_test,
                predictions
            )

            precision_value = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall_value = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1_value = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            st.subheader("Model Performance")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Accuracy",
                    f"{accuracy_value * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Precision",
                    f"{precision_value * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Recall",
                    f"{recall_value * 100:.2f}%"
                )

            with col4:
                st.metric(
                    "F1-Score",
                    f"{f1_value * 100:.2f}%"
                )

            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                y_test,
                predictions
            )

            fig, ax = plt.subplots(
                figsize=(6, 5)
            )

            ConfusionMatrixDisplay(
                confusion_matrix=cm
            ).plot(
                ax=ax
            )

            st.pyplot(fig)

            st.success(
                "Model evaluation completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Evaluation error: {error}"
            )


st.header("Multiple ML Models Comparison")

if uploaded_file is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="multiple_models_target"
    )

    if st.button("Train Multiple Models"):

        try:

            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                encoder = LabelEncoder()

                y = encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            models = {
                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        random_state=42,
                        max_depth=5
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
            }

            results = []

            for model_name, model_algorithm in models.items():

                model = Pipeline([
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        model_algorithm
                    )
                ])

                model.fit(
                    X_train,
                    y_train
                )

                predictions = model.predict(
                    X_test
                )

                model_accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                model_precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                model_recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                model_f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                results.append({
                    "Model": model_name,
                    "Accuracy": model_accuracy,
                    "Precision": model_precision,
                    "Recall": model_recall,
                    "F1-Score": model_f1
                })

            results_df = pd.DataFrame(
                results
            )

            st.subheader("Model Performance Comparison")

            st.dataframe(
                results_df.round(3),
                use_container_width=True
            )

            st.subheader("Performance Chart")

            chart_data = results_df.set_index(
                "Model"
            )

            st.bar_chart(
                chart_data[
                    [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-Score"
                    ]
                ]
            )

            st.success(
                "Multiple ML models trained successfully!"
            )

        except Exception as error:

            st.error(
                f"Model training error: {error}"
            )


st.header("Model Diagnosis")

if uploaded_file is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="diagnosis_target"
    )

    if st.button("Run Model Diagnosis"):

        try:

            from sklearn.model_selection import (
                train_test_split,
                StratifiedKFold,
                cross_val_score
            )

            from sklearn.preprocessing import (
                LabelEncoder,
                OneHotEncoder
            )

            from sklearn.compose import (
                ColumnTransformer
            )

            from sklearn.pipeline import (
                Pipeline
            )

            from sklearn.impute import (
                SimpleImputer
            )

            from sklearn.linear_model import (
                LogisticRegression
            )

            from sklearn.tree import (
                DecisionTreeClassifier
            )

            from sklearn.ensemble import (
                RandomForestClassifier
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                encoder = LabelEncoder()

                y = encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            models = {
                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        max_depth=5,
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
            }

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            diagnosis_results = []

            cv = StratifiedKFold(
                n_splits=3,
                shuffle=True,
                random_state=42
            )

            for model_name, algorithm in models.items():

                model = Pipeline([
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        algorithm
                    )
                ])

                model.fit(
                    X_train,
                    y_train
                )

                train_score = model.score(
                    X_train,
                    y_train
                )

                test_score = model.score(
                    X_test,
                    y_test
                )

                cv_scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring="accuracy"
                )

                cv_mean = cv_scores.mean()

                gap = train_score - test_score

                if gap > 0.20:

                    diagnosis = "Possible Overfitting"

                elif test_score < 0.60:

                    diagnosis = "Low Test Performance"

                else:

                    diagnosis = "Stable"

                diagnosis_results.append({
                    "Model": model_name,
                    "Train Accuracy": train_score,
                    "Test Accuracy": test_score,
                    "CV Accuracy": cv_mean,
                    "Train-Test Gap": gap,
                    "Diagnosis": diagnosis
                })

            diagnosis_df = pd.DataFrame(
                diagnosis_results
            )

            st.subheader(
                "Model Diagnosis Results"
            )

            st.dataframe(
                diagnosis_df.round(3),
                use_container_width=True
            )

            st.subheader(
                "Cross-Validation Accuracy"
            )

            cv_chart = diagnosis_df.set_index(
                "Model"
            )

            st.bar_chart(
                cv_chart[
                    ["CV Accuracy"]
                ]
            )

            for result in diagnosis_results:

                if result["Diagnosis"] == "Possible Overfitting":

                    st.warning(
                        f'{result["Model"]}: '
                        "Possible overfitting detected."
                    )

                elif result["Diagnosis"] == "Low Test Performance":

                    st.warning(
                        f'{result["Model"]}: '
                        "Low testing performance detected."
                    )

                else:

                    st.success(
                        f'{result["Model"]}: '
                        "Model performance looks stable."
                    )

            st.success(
                "Model diagnosis completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Diagnosis error: {error}"
            )


st.header("Feature Importance Analysis")

if uploaded_file is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="feature_target"
    )

    if st.button("Analyze Feature Importance"):

        try:

            from sklearn.preprocessing import (
                OneHotEncoder
            )

            from sklearn.compose import (
                ColumnTransformer
            )

            from sklearn.pipeline import (
                Pipeline
            )

            from sklearn.impute import (
                SimpleImputer
            )

            from sklearn.model_selection import (
                train_test_split
            )

            from sklearn.ensemble import (
                RandomForestClassifier
            )

            from sklearn.preprocessing import (
                LabelEncoder
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                encoder = LabelEncoder()

                y = encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            model = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                )
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model.fit(
                X_train,
                y_train
            )

            trained_model = model.named_steps[
                "model"
            ]

            trained_preprocessor = model.named_steps[
                "preprocessor"
            ]

            feature_names = (
                trained_preprocessor
                .get_feature_names_out()
            )

            importance_values = (
                trained_model
                .feature_importances_
            )

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importance_values
            })

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=False
            )

            st.subheader(
                "Feature Importance Results"
            )

            st.dataframe(
                importance_df.round(4),
                use_container_width=True
            )

            st.subheader(
                "Feature Importance Chart"
            )

            chart_data = (
                importance_df
                .head(10)
                .set_index("Feature")
            )

            st.bar_chart(
                chart_data["Importance"]
            )

            st.success(
                "Feature importance analysis completed successfully!"
            )

        except Exception as error:

            st.error(
                f"Feature analysis error: {error}"
            )


st.header("Automatic Improvement Suggestions")

if uploaded_file is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="suggestion_target"
    )

    if st.button("Generate Improvement Suggestions"):

        try:

            from sklearn.model_selection import (
                train_test_split
            )

            from sklearn.preprocessing import (
                LabelEncoder,
                OneHotEncoder
            )

            from sklearn.compose import (
                ColumnTransformer
            )

            from sklearn.pipeline import (
                Pipeline
            )

            from sklearn.impute import (
                SimpleImputer
            )

            from sklearn.linear_model import (
                LogisticRegression
            )

            from sklearn.metrics import (
                accuracy_score
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                encoder = LabelEncoder()

                y = encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            model = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model.fit(
                X_train,
                y_train
            )

            train_accuracy = model.score(
                X_train,
                y_train
            )

            test_accuracy = model.score(
                X_test,
                y_test
            )

            missing_values = int(
                data.isnull().sum().sum()
            )

            duplicate_rows = int(
                data.duplicated().sum()
            )

            train_test_gap = (
                train_accuracy -
                test_accuracy
            )

            st.subheader(
                "Model Analysis"
            )

            st.write(
                "Training Accuracy:",
                f"{train_accuracy * 100:.2f}%"
            )

            st.write(
                "Testing Accuracy:",
                f"{test_accuracy * 100:.2f}%"
            )

            st.write(
                "Missing Values:",
                missing_values
            )

            st.write(
                "Duplicate Rows:",
                duplicate_rows
            )

            st.subheader(
                "Improvement Suggestions"
            )

            suggestions = []

            if missing_values > 0:

                suggestions.append(
                    "Handle missing values using suitable imputation methods."
                )

            if duplicate_rows > 0:

                suggestions.append(
                    "Remove duplicate rows to improve data quality."
                )

            if train_test_gap > 0.20:

                suggestions.append(
                    "Possible overfitting detected. "
                    "Try regularization, simpler models, "
                    "or more training data."
                )

            if test_accuracy < 0.60:

                suggestions.append(
                    "Testing performance is low. "
                    "Try better features or different ML algorithms."
                )

            if len(numeric_features) == 0:

                suggestions.append(
                    "Consider creating useful numerical features "
                    "if appropriate for the dataset."
                )

            if len(suggestions) == 0:

                suggestions.append(
                    "Current model and dataset look reasonably stable. "
                    "Continue testing with new data."
                )

            for suggestion in suggestions:

                st.info(
                    " " + suggestion
                )

            st.success(
                "Improvement suggestions generated successfully!"
            )

        except Exception as error:

            st.error(
                f"Suggestion error: {error}"
            )


st.header("Complete Analysis Report")

if uploaded_file is not None:

    if st.button("Generate Complete Report"):

        try:

            report_data = []

            report_data.append({
                "Category": "Dataset",
                "Metric": "Rows",
                "Value": df.shape[0]
            })

            report_data.append({
                "Category": "Dataset",
                "Metric": "Columns",
                "Value": df.shape[1]
            })

            report_data.append({
                "Category": "Data Quality",
                "Metric": "Missing Values",
                "Value": int(
                    df.isnull().sum().sum()
                )
            })

            report_data.append({
                "Category": "Data Quality",
                "Metric": "Duplicate Rows",
                "Value": int(
                    df.duplicated().sum()
                )
            })

            numeric_columns = df.select_dtypes(
                include=np.number
            ).columns

            total_outliers = 0

            for column in numeric_columns:

                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)

                IQR = Q3 - Q1

                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR

                outliers = df[
                    (df[column] < lower_limit) |
                    (df[column] > upper_limit)
                ]

                total_outliers += len(outliers)

            report_data.append({
                "Category": "Data Quality",
                "Metric": "Outliers",
                "Value": total_outliers
            })

            quality_score = 100

            if df.isnull().sum().sum() > 0:
                quality_score -= 20

            if df.duplicated().sum() > 0:
                quality_score -= 20

            if total_outliers > 0:
                quality_score -= 10

            quality_score = max(
                quality_score,
                0
            )

            report_data.append({
                "Category": "Data Quality",
                "Metric": "Quality Score",
                "Value": quality_score
            })

            report_df = pd.DataFrame(
                report_data
            )

            st.subheader(
                "Complete Report"
            )

            st.dataframe(
                report_df,
                use_container_width=True
            )

            report_csv = report_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Complete Report",
                data=report_csv,
                file_name="AI_Data_Detective_Report.csv",
                mime="text/csv"
            )

            st.success(
                "Complete analysis report generated successfully!"
            )

        except Exception as error:

            st.error(
                f"Report generation error: {error}"
            )


st.header("New Data Prediction")

if uploaded_file is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="prediction_target"
    )

    if st.button("Prepare Prediction Model"):

        try:

            from sklearn.preprocessing import (
                LabelEncoder,
                OneHotEncoder
            )

            from sklearn.compose import (
                ColumnTransformer
            )

            from sklearn.pipeline import (
                Pipeline
            )

            from sklearn.impute import (
                SimpleImputer
            )

            from sklearn.linear_model import (
                LogisticRegression
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            if y.dtype == "object":

                target_encoder = LabelEncoder()

                y = target_encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            prediction_model = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ])

            prediction_model.fit(
                X,
                y
            )

            st.session_state[
                "prediction_model"
            ] = prediction_model

            st.session_state[
                "prediction_features"
            ] = X.columns.tolist()

            if y.dtype == "object":

                st.session_state[
                    "target_encoder"
                ] = target_encoder

            st.success(
                "Prediction model prepared successfully!"
            )

        except Exception as error:

            st.error(
                f"Model preparation error: {error}"
            )

    if "prediction_model" in st.session_state:

        st.subheader(
            "Enter New Data"
        )

        prediction_input = {}

        feature_columns = (
            st.session_state[
                "prediction_features"
            ]
        )

        for column in feature_columns:

            if pd.api.types.is_numeric_dtype(
                df[column]
            ):

                default_value = float(
                    df[column].median()
                )

                prediction_input[column] = st.number_input(
                    column,
                    value=default_value,
                    key=f"prediction_{column}"
                )

            else:

                values = (
                    df[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if len(values) > 0:

                    prediction_input[column] = st.selectbox(
                        column,
                        values,
                        key=f"prediction_{column}"
                    )

        if st.button(
            "Predict New Data"
        ):

            try:

                new_data = pd.DataFrame([
                    prediction_input
                ])

                model = st.session_state[
                    "prediction_model"
                ]

                prediction = model.predict(
                    new_data
                )

                result = prediction[0]

                if "target_encoder" in st.session_state:

                    encoder = st.session_state[
                        "target_encoder"
                    ]

                    result = encoder.inverse_transform(
                        [result]
                    )[0]

                st.subheader(
                    "Prediction Result"
                )

                st.success(
                    f"Predicted Result: {result}"
                )

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            new_data
                        )
                    )

                    confidence = (
                        probabilities.max()
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence * 100:.2f}%"
                    )

            except Exception as error:

                st.error(
                    f"Prediction error: {error}"
                )


st.header("Save Trained Model")

if uploaded_file is not None:

    target_column = st.selectbox(
        "Select Target Column",
        df.columns,
        key="save_model_target"
    )

    if st.button("Train and Save Model"):

        try:

            import joblib

            from sklearn.preprocessing import (
                LabelEncoder,
                OneHotEncoder
            )

            from sklearn.compose import (
                ColumnTransformer
            )

            from sklearn.pipeline import (
                Pipeline
            )

            from sklearn.impute import (
                SimpleImputer
            )

            from sklearn.linear_model import (
                LogisticRegression
            )

            data = df.copy()

            X = data.drop(
                target_column,
                axis=1
            )

            y = data[target_column]

            target_encoder = None

            if y.dtype == "object":

                target_encoder = LabelEncoder()

                y = target_encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            final_model = Pipeline([
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000
                    )
                )
            ])

            final_model.fit(
                X,
                y
            )

            joblib.dump(
                final_model,
                "trained_model.pkl"
            )

            if target_encoder is not None:

                joblib.dump(
                    target_encoder,
                    "target_encoder.pkl"
                )

            st.success(
                "Trained model saved successfully!"
            )

            with open(
                "trained_model.pkl",
                "rb"
            ) as model_file:

                st.download_button(
                    label="Download Trained Model",
                    data=model_file,
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )

            if target_encoder is not None:

                with open(
                    "target_encoder.pkl",
                    "rb"
                ) as encoder_file:

                    st.download_button(
                        label="Download Target Encoder",
                        data=encoder_file,
                        file_name="target_encoder.pkl",
                        mime="application/octet-stream"
                    )

        except Exception as error:

            st.error(
                f"Model saving error: {error}"
            )

st.header("Load Saved Model")

if uploaded_file is not None:

    if st.button("Load Saved Model"):

        try:

            import joblib
            import os

            if os.path.exists("trained_model.pkl"):

                loaded_model = joblib.load(
                    "trained_model.pkl"
                )

                st.session_state["loaded_model"] = loaded_model

                st.success(
                    "Saved model loaded successfully!"
                )

            else:

                st.warning(
                    "trained_model.pkl not found. "
                    "Please train and save the model first."
                )

        except Exception as error:

            st.error(
                f"Model loading error: {error}"
            )

if "loaded_model" in st.session_state:

    st.subheader("Make Prediction Using Saved Model")

    model = st.session_state["loaded_model"]

    feature_columns = [
        column
        for column in df.columns
        if column != "Placement"
    ]

    input_data = {}

    for column in feature_columns:

        if pd.api.types.is_numeric_dtype(
            df[column]
        ):

            default_value = float(
                df[column].median()
            )

            input_data[column] = st.number_input(
                f"Enter {column}",
                value=default_value,
                key=f"loaded_{column}"
            )

        else:

            options = df[column].dropna().unique().tolist()

            if len(options) > 0:

                input_data[column] = st.selectbox(
                    f"Select {column}",
                    options,
                    key=f"loaded_{column}"
                )

    if st.button("Predict Using Saved Model"):

        try:

            new_data = pd.DataFrame(
                [input_data]
            )

            prediction = model.predict(
                new_data
            )

            st.subheader("Prediction Result")

            st.success(
                f"Predicted Result: {prediction[0]}"
            )

            if hasattr(
                model,
                "predict_proba"
            ):

                probability = model.predict_proba(
                    new_data
                )

                confidence = probability.max() * 100

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

        except Exception as error:

            st.error(
                f"Prediction error: {error}"
            )

st.header("Batch Prediction")

if "loaded_model" in st.session_state:

    batch_file = st.file_uploader(
        "Upload CSV for Batch Prediction",
        type=["csv"],
        key="batch_prediction_file"
    )

    if batch_file is not None:

        try:

            batch_df = pd.read_csv(
                batch_file
            )

            st.subheader(
                "Uploaded Prediction Data"
            )

            st.dataframe(
                batch_df,
                use_container_width=True
            )

            if st.button(
                "Run Batch Prediction"
            ):

                model = st.session_state[
                    "loaded_model"
                ]

                predictions = model.predict(
                    batch_df
                )

                result_df = batch_df.copy()

                result_df[
                    "Predicted Result"
                ] = predictions

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            batch_df
                        )
                    )

                    result_df[
                        "Prediction Confidence (%)"
                    ] = (
                        probabilities.max(
                            axis=1
                        ) * 100
                    ).round(2)

                st.success(
                    "Batch prediction completed successfully!"
                )

                st.subheader(
                    "Batch Prediction Results"
                )

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                csv_result = result_df.to_csv(
                    index=False
                )

                st.download_button(
                    label="Download Batch Prediction Results",
                    data=csv_result,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as error:

            st.error(
                f"Batch prediction error: {error}"
            )

else:

    st.info(
        "Please load the saved model before running batch prediction."
    )

st.header("Prediction Summary")

if "batch_prediction_file" in st.session_state:

    batch_file = st.session_state["batch_prediction_file"]

if "loaded_model" in st.session_state:

    summary_file = st.file_uploader(
        "Upload CSV for Prediction Summary",
        type=["csv"],
        key="prediction_summary_file"
    )

    if summary_file is not None:

        try:

            summary_df = pd.read_csv(
                summary_file
            )

            model = st.session_state[
                "loaded_model"
            ]

            predictions = model.predict(
                summary_df
            )

            result_df = summary_df.copy()

            result_df[
                "Predicted Result"
            ] = predictions

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    summary_df
                )

                result_df[
                    "Confidence (%)"
                ] = (
                    probabilities.max(
                        axis=1
                    ) * 100
                )

            total_predictions = len(
                result_df
            )

            result_counts = result_df[
                "Predicted Result"
            ].value_counts()

            st.subheader(
                "Prediction Overview"
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Total Predictions",
                    total_predictions
                )

            with col2:
                st.metric(
                    "Most Predicted Result",
                    result_counts.index[0]
                )

            with col3:

                if "Confidence (%)" in result_df.columns:

                    average_confidence = (
                        result_df[
                            "Confidence (%)"
                        ].mean()
                    )

                    st.metric(
                        "Average Confidence",
                        f"{average_confidence:.2f}%"
                    )

            st.subheader(
                "Prediction Distribution"
            )

            st.bar_chart(
                result_counts
            )

            st.subheader(
                "Detailed Prediction Report"
            )

            st.dataframe(
                result_df,
                use_container_width=True
            )

            report_csv = result_df.to_csv(
                index=False
            )

            st.download_button(
                label="Download Prediction Report",
                data=report_csv,
                file_name="prediction_report.csv",
                mime="text/csv"
            )

        except Exception as error:

            st.error(
                f"Prediction summary error: {error}"
            )

else:

    st.info(
        "Please load the saved model first."
    )

st.header("Model Performance Dashboard")

if uploaded_file is not None:

    performance_target = st.selectbox(
        "Select Target Column",
        df.columns,
        key="performance_target"
    )

    if st.button(
        "Generate Model Performance"
    ):

        try:

            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import (
                LabelEncoder,
                OneHotEncoder
            )
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score
            )

            data = df.copy()

            X = data.drop(
                performance_target,
                axis=1
            )

            y = data[performance_target]

            if y.dtype == "object":

                encoder = LabelEncoder()

                y = encoder.fit_transform(y)

            numeric_features = X.select_dtypes(
                include=np.number
            ).columns.tolist()

            categorical_features = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            numeric_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ])

            categorical_pipeline = Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ])

            preprocessor = ColumnTransformer([
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                )
            ])

            models = {
                "Logistic Regression":
                    LogisticRegression(
                        max_iter=1000
                    ),

                "Decision Tree":
                    DecisionTreeClassifier(
                        max_depth=5,
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
            }

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            results = []

            for model_name, algorithm in models.items():

                pipeline = Pipeline([
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        algorithm
                    )
                ])

                pipeline.fit(
                    X_train,
                    y_train
                )

                predictions = pipeline.predict(
                    X_test
                )

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(
                        y_test,
                        predictions
                    ),
                    "Precision": precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "F1-Score": f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                })

            performance_df = pd.DataFrame(
                results
            )

            st.subheader(
                "Model Performance Results"
            )

            st.dataframe(
                performance_df.round(3),
                use_container_width=True
            )

            st.subheader(
                "Performance Comparison"
            )

            chart_data = performance_df.set_index(
                "Model"
            )

            st.bar_chart(
                chart_data[
                    [
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1-Score"
                    ]
                ]
            )

            st.subheader(
                "Performance Metrics"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Best Accuracy",
                    f"{performance_df['Accuracy'].max() * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Best Precision",
                    f"{performance_df['Precision'].max() * 100:.2f}%"
                )

            with col3:
                st.metric(
                    "Best Recall",
                    f"{performance_df['Recall'].max() * 100:.2f}%"
                )

            with col4:
                st.metric(
                    "Best F1-Score",
                    f"{performance_df['F1-Score'].max() * 100:.2f}%"
                )

        except Exception as error:

            st.error(
                f"Performance analysis error: {error}"
            )

else:

    st.info(
        "Please upload a CSV dataset first."
    )

st.header("Final AI Data Detective Report")

if uploaded_file is not None:

    if st.button(
        "Generate Final Project Report"
    ):

        try:

            data = df.copy()

            total_rows = data.shape[0]
            total_columns = data.shape[1]

            missing_values = int(
                data.isnull().sum().sum()
            )

            duplicate_rows = int(
                data.duplicated().sum()
            )

            numeric_columns = data.select_dtypes(
                include=np.number
            ).columns.tolist()

            total_outliers = 0

            for column in numeric_columns:

                Q1 = data[column].quantile(0.25)
                Q3 = data[column].quantile(0.75)

                IQR = Q3 - Q1

                lower_limit = (
                    Q1 - 1.5 * IQR
                )

                upper_limit = (
                    Q3 + 1.5 * IQR
                )

                outlier_count = (
                    (
                        data[column] < lower_limit
                    )
                    |
                    (
                        data[column] > upper_limit
                    )
                ).sum()

                total_outliers += int(
                    outlier_count
                )

            quality_score = 100

            if missing_values > 0:
                quality_score -= 20

            if duplicate_rows > 0:
                quality_score -= 20

            if total_outliers > 0:
                quality_score -= 10

            quality_score = max(
                quality_score,
                0
            )

            report_data = {
                "Report": [
                    "AI Data Detective & Model Doctor"
                ],
                "Dataset Rows": [
                    total_rows
                ],
                "Dataset Columns": [
                    total_columns
                ],
                "Missing Values": [
                    missing_values
                ],
                "Duplicate Rows": [
                    duplicate_rows
                ],
                "Detected Outliers": [
                    total_outliers
                ],
                "Data Quality Score": [
                    quality_score
                ]
            }

            final_report = pd.DataFrame(
                report_data
            )

            st.subheader(
                "Final Dataset Analysis"
            )

            st.dataframe(
                final_report,
                use_container_width=True
            )

            st.subheader(
                "Project Findings"
            )

            if missing_values == 0:

                st.success(
                    "No missing values detected."
                )

            else:

                st.warning(
                    f"{missing_values} missing values detected."
                )

            if duplicate_rows == 0:

                st.success(
                    "No duplicate rows detected."
                )

            else:

                st.warning(
                    f"{duplicate_rows} duplicate rows detected."
                )

            if total_outliers == 0:

                st.success(
                    "No numerical outliers detected."
                )

            else:

                st.warning(
                    f"{total_outliers} numerical outliers detected."
                )

            if quality_score >= 80:

                st.success(
                    f"Data Quality Score: {quality_score}/100"
                )

            elif quality_score >= 60:

                st.warning(
                    f"Data Quality Score: {quality_score}/100"
                )

            else:

                st.error(
                    f"Data Quality Score: {quality_score}/100"
                )

            st.subheader(
                "Final Recommendations"
            )

            recommendations = []

            if missing_values > 0:

                recommendations.append(
                    "Handle missing values before machine learning."
                )

            if duplicate_rows > 0:

                recommendations.append(
                    "Remove duplicate records."
                )

            if total_outliers > 0:

                recommendations.append(
                    "Review numerical outliers before model training."
                )

            if not recommendations:

                recommendations.append(
                    "Dataset quality is acceptable for further analysis."
                )

            for recommendation in recommendations:

                st.info(
                    recommendation
                )

            report_csv = final_report.to_csv(
                index=False
            )

            st.download_button(
                label="Download Final Project Report",
                data=report_csv,
                file_name="AI_Data_Detective_Final_Report.csv",
                mime="text/csv"
            )

            st.success(
                "Final project report generated successfully!"
            )

        except Exception as error:

            st.error(
                f"Report generation error: {error}"
            )

else:

    st.info(
        "Please upload a CSV dataset first."
    )

st.sidebar.title("AI Data Detective")

st.sidebar.write(
    "Data Quality and Machine Learning Analysis"
)

st.sidebar.markdown(
    """
    ### Project Modules

     Data Detective

     Data Cleaning

     Data Visualization

     Model Training

     Model Evaluation

     Model Diagnosis

     Improvement Suggestions

     Prediction

     Final Report
    """
)

st.sidebar.info(
    "AI Data Detective & Model Doctor"
)

st.sidebar.caption(
    "Developed using Python and Machine Learning"
)
