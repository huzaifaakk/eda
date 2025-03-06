import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set style for matplotlib plots
plt.style.use('ggplot')
sns.set(style="whitegrid")


def load_data():
    file_path = r"D:\Titanic-Dataset.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def display_basic_info(df):
    """Display basic information about the dataset"""
    print("\n=== DATASET OVERVIEW ===")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nSummary statistics:")
    print(df.describe())

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_percent
    })
    print(missing_data[missing_data['Missing Values'] > 0])


def analyze_survival(df):
    """Analyze survival rates"""
    print("\n=== SURVIVAL ANALYSIS ===")

    # Overall survival rate
    survival_rate = df['Survived'].mean() * 100
    print(f"\nOverall survival rate: {survival_rate:.2f}%")

    
    gender_survival = df.groupby('Sex')['Survived'].mean() * 100
    print("\nSurvival rate by gender:")
    print(gender_survival)

    class_survival = df.groupby('Pclass')['Survived'].mean() * 100
    print("\nSurvival rate by passenger class:")
    print(class_survival)

    port_survival = df.groupby('Embarked')['Survived'].mean() * 100
    print("\nSurvival rate by embarkation port:")
    print(port_survival)


def visualize_data(df):
    """Create visualizations for the dataset"""
    print("\n=== CREATING VISUALIZATIONS ===")
    print("Generating plots...")

    #directory for plots
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Survived', data=df, palette='Blues')
    plt.title('Survival Distribution', fontsize=16)
    plt.xlabel('Survived (0 = No, 1 = Yes)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig('plots/survival_distribution.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sex', hue='Survived', data=df, palette='Blues')
    plt.title('Survival by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('plots/survival_by_gender.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='Blues')
    plt.title('Survival by Passenger Class', fontsize=16)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('plots/survival_by_class.png')
    plt.show()

    # 4. Age distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=20, palette='Blues')
    plt.title('Age Distribution by Survival', fontsize=16)
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Survived', labels=['No', 'Yes'])
    plt.savefig('plots/age_distribution.png')
    plt.show()


def analyze_age_groups(df):
    """Analyze survival rates by age groups"""
    bins = [0, 10, 20, 30, 40, 50, 60, float('inf')]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    plt.figure(figsize=(8, 5))
    age_survival = df.groupby('AgeGroup')['Survived'].mean() * 100
    age_survival.plot(kind='bar', color='skyblue')

    plt.title("Survival Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Survival Rate (%)")
    plt.xticks(rotation=45)

    plt.savefig('plots/survival_by_age_group.png')
    plt.show()


def feature_importance(df):
    """Calculate feature importance for survival prediction"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    model_df = df.copy()
    model_df['Age'].fillna(model_df['Age'].median(), inplace=True)
    model_df['Embarked'].fillna(model_df['Embarked'].mode()[0], inplace=True)
    model_df.drop('Cabin', axis=1, inplace=True)

    le = LabelEncoder()
    model_df['Sex'] = le.fit_transform(model_df['Sex'])
    model_df['Embarked'] = le.fit_transform(model_df['Embarked'])

    model_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

    X = model_df.drop('Survived', axis=1)
    y = model_df['Survived']

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importance = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
    importance.sort_values(by='Importance', ascending=False, inplace=True)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='Blues')
    plt.title('Feature Importance for Survival Prediction', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.savefig('plots/feature_importance.png')
    plt.show()


def main():
    """Main function to run the EDA"""
    print("=== TITANIC DATASET EXPLORATORY DATA ANALYSIS ===")

    df = load_data()
    if df is None:
        return

    display_basic_info(df)
    analyze_survival(df)
    analyze_age_groups(df)
    visualize_data(df)
    feature_importance(df)

    print("\n=== EDA COMPLETED SUCCESSFULLY ===")
    print("Check the 'plots' directory for saved visualizations.")


if __name__ == "__main__":
    main()
