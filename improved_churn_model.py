import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from lifelines import CoxPHFitter
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImprovedChurnModel:
    def __init__(self):
        self.preprocessor = None
        self.classifier = None
        self.cox_model = None
        self.feature_names = None
        self.numerical_cols = None
        self.categorical_cols = None

    def preprocess_data(self, df, missing_threshold=80, zero_threshold=80):
        logging.info("Preprocessing data...")
        
        # List of important features that should not be removed
        important_features = ['lost_program', 'days_rem_to_end_contract', 'startDate', 'endDate', 'last_activity_date']
        
        # Remove data leakage columns
        leakage_columns = ['churn_date']
        id_columns = ['datalakeProgramId']
        date_columns = [col for col in df.columns if 'date' in col.lower() and col not in important_features]
        
        drop_columns = leakage_columns + id_columns + date_columns
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        
        # Convert categorical columns
        categorical_columns = ['siteCenter', 'customerSize']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Remove columns with high missing values
        missing_values = df.isnull().mean() * 100
        high_missing_cols = [col for col in missing_values[missing_values > missing_threshold].index if col not in important_features]
        df = df.drop(columns=high_missing_cols)
        
        # Remove columns with high zero values
        zero_values = (df == 0).mean() * 100
        high_zero_cols = [col for col in zero_values[zero_values > zero_threshold].index if col not in important_features]
        df = df.drop(columns=high_zero_cols)
        
        logging.info(f"Removed {len(high_missing_cols)} columns with >{missing_threshold}% missing values")
        logging.info(f"Removed {len(high_zero_cols)} columns with >{zero_threshold}% zero values")
        logging.info(f"Final dataset shape: {df.shape}")
        
        return df

    def prepare_survival_data(self, df):
        if 'days_rem_to_end_contract' in df.columns:
            df['duration'] = df['days_rem_to_end_contract']
        elif 'endDate' in df.columns and 'startDate' in df.columns:
            df['startDate'] = pd.to_datetime(df['startDate'])
            df['endDate'] = pd.to_datetime(df['endDate'])
            df['duration'] = (df['endDate'] - df['startDate']).dt.days
        else:
            raise ValueError("No suitable duration column found for survival analysis")
        
        df['event'] = df['lost_program']
        return df

    def fit(self, X, y):
        logging.info("Fitting the model...")
        
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['category']).columns.tolist()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), self.categorical_cols)
            ])
        
        self.classifier = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ])
        
        self.classifier.fit(X, y)
        
        # Prepare feature names
        self.feature_names = (self.numerical_cols + 
                              self.preprocessor.named_transformers_['cat']
                              .get_feature_names_out(self.categorical_cols).tolist())
        
        # Fit Cox model
        cox_df = self.prepare_survival_data(X.join(y))
        cox_features = [col for col in self.numerical_cols if col not in ['duration', 'event']]
        cox_df = cox_df[cox_features + ['duration', 'event']]
        cox_df = cox_df.replace([np.inf, -np.inf], np.nan)
        
        imputer = SimpleImputer(strategy='median')
        cox_df[cox_features] = imputer.fit_transform(cox_df[cox_features])
        
        scaler = StandardScaler()
        cox_df[cox_features] = scaler.fit_transform(cox_df[cox_features])
        
        self.cox_model = CoxPHFitter()
        self.cox_model.fit(cox_df, duration_col='duration', event_col='event')

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)[:, 1]

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def get_feature_importance(self):
        classifier = self.classifier.named_steps['classifier']
        importance = classifier.feature_importances_
        return sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True)

    def get_cox_summary(self):
        return self.cox_model.summary

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

def main():
    # Load data
    df = pd.read_csv('../Churn/features_extracted.csv')
    
    # Create model instance
    model = ImprovedChurnModel()
    
    # Preprocess data
    df_processed = model.preprocess_data(df)
    
    # Split data
    X = df_processed.drop(columns=['lost_program'])
    y = df_processed['lost_program']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Get feature importance
    importance = model.get_feature_importance()
    print("\nTop 10 Features by Importance:")
    for feature, imp in importance[:10]:
        print(f"{feature}: {imp:.4f}")
    
    # Get Cox model summary
    cox_summary = model.get_cox_summary()
    print("\nCox Model Summary (top 10 features):")
    print(cox_summary.sort_values('p').head(10))
    
    # Save model
    model.save_model('improved_churn_model.joblib')
    print("\nModel saved as 'improved_churn_model.joblib'")

if __name__ == "__main__":
    main()