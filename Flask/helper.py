import pandas as pd
import numpy as np
import joblib
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# for clinic data
def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def feature_extraction(df):

    # Glucose level
    glucose_bins = [0, 70, 100, float('inf')]
    glucose_labels = ['Low', 'Normal', 'High']

    df['Glucose_Level'] = pd.cut(df['Glucose'], bins=glucose_bins, labels=glucose_labels)

    # BMI category
    bmi_bins = [0, 18.5, 25, 30, 35, 40, float('inf')]
    bmi_labels = ['underweight', 'normal', 'overweight', 'obese class I', 'obese class II', 'obese class III']
    df['bmi_category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels)

    # Create a new column with the interaction between glucose and BMI
    df['Glucose and BMI Interaction'] = df['Glucose'] * df['BMI']

    # Create a new column with blood sugar level category
    bsl_ranges = [0, 80, 140, np.inf]
    bsl_labels = ['Normal', 'High', 'Very High']

    df['Blood Sugar Level'] = pd.cut(df['Glucose'], bins=bsl_ranges, labels=bsl_labels)

    # Create a new feature for insulin category
    df['Insulin_Category'] = pd.cut(df['Insulin'], bins=[-1, 50, 200, 1000], labels=['Low', 'Normal', 'High'])

    # Create age and BMI interaction feature
    df["Age_BMI_Interaction"] = df["Age"] * df["BMI"]

    # Age Difference
    average_age = df['Age'].mean()
    df['Age_Difference'] = df['Age'] - average_age

    # Create diabetes history feature
    dpf_threshold = 0.8
    df["Diabetes_History"] = df["DiabetesPedigreeFunction"].apply(lambda x: "Yes" if x >= dpf_threshold else "No")

    # Calculate Insulin-Glucose Ratio feature
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / df["Glucose"]

    # Risk score
    features = ['Glucose', 'BMI', 'Age', 'Insulin', 'BloodPressure']
    weights = [0.2, 0.25, 0.2, 0.15, 0.2]

    df['risk_score'] = (df[features] * weights).sum(axis=1)

    # Glucose Level Classification
    df['Glucose_Level'] = pd.cut(df['Glucose'], bins=[0, 70, 99, float('inf')],
                                labels=['Low Glucose', 'Normal Glucose', 'High Glucose'])

    # Insulin Level Classification
    df['Insulin_Level'] = pd.cut(df['Insulin'], bins=[0, 30, 199, float('inf')],
                                labels=['Low Insulin', 'Normal Insulin', 'High Insulin'])

    # Mean Encoding
    mean_encoding = df.groupby('BMI')['Outcome'].mean()
    df['BMI_Category_Mean_Encoding'] = df['BMI'].map(mean_encoding)

    # For insulin
    mean_encoding = df.groupby('Insulin')['Outcome'].mean()
    df['Insulin_Level_Mean_Encoding'] = df['Insulin'].map(mean_encoding)

    # For glucose
    mean_encoding = df.groupby('Glucose')['Outcome'].mean()
    df['Glucose_Level_Mean_Encoding'] = df['Glucose'].map(mean_encoding)

    # For blood pressure
    mean_encoding = df.groupby('BloodPressure')['Outcome'].mean()
    df['BloodPressure_Level_Mean_Encoding'] = df['BloodPressure'].map(mean_encoding)

    # For diabetes pedigree function
    mean_encoding = df.groupby('DiabetesPedigreeFunction')['Outcome'].mean()
    df['DiabetesPedigreeFunction_Level_Mean_Encoding'] = df['DiabetesPedigreeFunction'].map(mean_encoding)

    # For age
    mean_encoding = df.groupby('Age')['Outcome'].mean()
    df['Age_Level_Mean_Encoding'] = df['Age'].map(mean_encoding)

    # For skin thickness
    mean_encoding = df.groupby('SkinThickness')['Outcome'].mean()
    df['SkinThickness_Level_Mean_Encoding'] = df['SkinThickness'].map(mean_encoding)

    return df

def data_preprocessing(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                    BMI, DiabetesPedigreeFunction, Age):
    
    df = pd.read_csv("diabetes_clean.csv")
    
    df = df.append({"Pregnancies":Pregnancies, 
                    "Glucose":Glucose, 
                    "BloodPressure":BloodPressure, 
                    "SkinThickness":SkinThickness, 
                    "Insulin":Insulin, 
                    "BMI" : BMI, 
                    "DiabetesPedigreeFunction":DiabetesPedigreeFunction, 
                    "Age":Age,
                    "Outcome":np.nan},ignore_index=True)
    
    df = df[['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Outcome']]
    
    df = feature_extraction(df)
    
    cat_cols,num_cols, cat_but_car = grab_col_names(df)
    cat_cols = [col for col in cat_cols if col not in ["Outcome"]]
    
    df = one_hot_encoder(df, cat_cols, drop_first=True)
    df = df.iloc[[-1]]
    scaler = joblib.load("scaler.h5")
    df[num_cols] = scaler.transform(df[num_cols].iloc[[-1]])

    return df

def make_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                    BMI, DiabetesPedigreeFunction, Age):
    
    X = data_preprocessing(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, 
                    BMI, DiabetesPedigreeFunction, Age)
    
    
    
    arr = X.values
    arr = np.delete(arr, 8)
    
    ml_model = joblib.load("ml_model94%.h5")
    prediction = ml_model.predict([arr])
    probabilty = f"{int(round(np.max(ml_model.predict_proba([arr])[0]) * 100,0))} %"

    advice = make_advise(Glucose, BloodPressure, Insulin, BMI)

    if prediction[0] == 1:
        return True, probabilty, advice
    elif prediction[0] == 0:
        return False, probabilty, advice
    
def make_advise(Glucose, BloodPressure, Insulin, 
                    BMI):
    df = pd.DataFrame({ 
                    "Gula_darah":Glucose, 
                    "Tekanan_darah":BloodPressure,  
                    "Insulin":Insulin, 
                    "Berat_badan" : BMI,  
                    }, index=[0])
    sc = joblib.load("scaler2.joblib")

    X = sc.transform(df)

    weights = [11.27808751,  0.61907744,  4.57064833,  8.28770817]

    mult_arr = np.multiply(X[0], weights)

    if np.max(mult_arr) < 0:
        return "Kerja bagus"

    # significan features
    sig_ftrs = df.columns[np.argmax(mult_arr)]
    sig_ftrs = sig_ftrs.replace("_", " ")
    
    return f"{sig_ftrs} anda tinggi, mohon kurangi {sig_ftrs.lower()} anda!"