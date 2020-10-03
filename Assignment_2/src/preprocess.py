"""
 Assignment 2: Neural Networks
 Authors: Vignesh Vasan & Ronak Hegde
 Objective: Contains all required preprocessing for our dataset (handling missing values, one-hot encoding, binning, min-max scaling)
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def drop_rows(df, categorical_cols):

  ## drop rows where gender is Unknown/Invalid
  df.drop(df[df['gender'] == "Unknown/Invalid"].index, axis=0, inplace=True)

  ## dropping columns that have many missing values
  dropped_columns = ['weight', 'payer_code', 'medical_specialty']
  dropped_columns.append("encounter_id")
  dropped_columns.append('discharge_disposition_id')

  ## dropping columns that have little to no variability
  for col in categorical_cols:
      if(df[col].value_counts(normalize=True).max() > 0.948):
          dropped_columns.append(col)
          
  df.drop(columns=dropped_columns, axis=1, inplace=True)
  df.dropna(inplace=True)

  return df

def categorical_to_numerical(df):
  #converting our categorical variable 
  df['readmitted'] = np.where(df['readmitted']!='NO',1,0)

  new_ages = {
  "[0-10)": 5,
  "[10-20)": 15,
  "[20-30)": 25,
  "[30-40)": 35,
  "[40-50)": 45,
  "[50-60)": 55,
  "[60-70)": 65,
  "[70-80)": 75,
  "[80-90)": 85,
  "[90-100)": 95
  }

  max_glu_serums = {
  "None": 0,
  "Norm": 100,
  ">200": 200,
  ">300": 300
  }

  A1CResult_map = {
  "None": 0,
  "Norm": 5,
  ">7": 7,
  ">8": 8
  } 

  drug_codes = {
  "No": -20,
  "Down": -10, 
  "Steady": 0,
  "Up": 10    
  }

  df['age'] = df['age'].map(new_ages)
  df['max_glu_serum'] = df['max_glu_serum'].map(max_glu_serums)
  df['A1Cresult'] = df['A1Cresult'].map(A1CResult_map)

  drugs = ['metformin','glipizide','glyburide', 'pioglitazone', 'rosiglitazone','insulin'] 
  for drug in drugs:
    df[drug] = df[drug].map(drug_codes)

  #converting binary variables
  df['change'] = np.where(df['change']=='No',-1,1)
  df['diabetesMed'] = np.where(df['diabetesMed']=='No',-1,1)

  return df

def map_diagnosis(df):
  """ The three diagnosis columns have 700+ possible values which would result in 700+ columns
      with one-hot encoding. This reduces it to just nine by utilizing the mapping within the 
      research paper associated with this dataset"""

  diagnosis_cols = ['diag_1', 'diag_2', 'diag_3']

  for col in diagnosis_cols:
    df['tmp'] = np.nan
    df.loc[(df[col].str.contains("250")), col] = '250'
    df.loc[(df[col].str.startswith('V')) | (df[col].str.startswith('E')), col] = '-999' 

    df[col] = df[col].astype(float)
    
    #convert the correct ranges based on values given in paper
    df.loc[(((df[col] >=390) & (df[col]<=460)) | (df[col] == 785)), 'tmp'] = 'Circulatory'
    df.loc[(((df[col] >=460) & (df[col]<=519)) | (df[col] == 786)), 'tmp'] = 'Respiratory'
    df.loc[(((df[col] >=520) & (df[col]<=579)) | (df[col] == 787)), 'tmp'] = 'Digestive'
    df.loc[(((df[col] >=580) & (df[col]<=629)) | (df[col] == 788)), 'tmp'] = 'Genitourinary'
    df.loc[((df[col] >=800) & (df[col]<=999)), 'tmp'] = 'Injury'
    df.loc[((df[col] >=710) & (df[col]<=739)), 'tmp'] = 'Musculoskeletal'
    df.loc[((df[col] >=140) & (df[col]<=239)), 'tmp'] = 'Neoplasms'
    df.loc[(df[col] == 250), 'tmp'] = 'Diabetes'
    
    df['tmp'].fillna(value = "Other", inplace=True)
    
    df[col] = df['tmp']
    df.drop(columns=['tmp'], inplace=True)

  return df


def map_admissions(df):
  """reduce both admission column categories drastically by grouping similar nominal values together""" 

  df['tmp'] = np.nan
  col = 'admission_source_id'
  df.loc[((df[col].between(4,6)) | (df[col] == 10) | (df[col] == 18) | (df[col] == 22) | (df[col].between(25,26))), 'tmp'] = "Transfer_Source"
  df.loc[df[col].between(1,3), 'tmp'] = "Referral_Source"
  df.loc[((df[col].between(11,14))| (df[col].between(23,24))), 'tmp'] = "Birth_Source"
  df.loc[df[col] == 7, 'tmp'] = "Emergency_Source"
  df.loc[((df[col] == 8) | (df[col]==19)), 'tmp'] = "Other"
          
  df['tmp'].fillna(value = "Unknown", inplace=True)
  df[col] = df['tmp']
  df.drop(columns=['tmp'], inplace=True)


  ##mapping admission type_id
  df['tmp'] = np.nan
  col = 'admission_type_id'
  df.loc[df[col] == 1, 'tmp'] = 'Emergency_Type'
  df.loc[df[col] == 2, 'tmp'] = 'Urgent_Type'
  df.loc[df[col] == 3, 'tmp'] = 'Elective_Type'
  df.loc[df[col] == 7, 'tmp'] = 'Trauma_Type'
  df.loc[df[col] == 4, 'tmp'] = 'Newborn_Type'

  df['tmp'].fillna(value = "Unknown", inplace=True)
  df[col] = df['tmp']
  df.drop(columns=['tmp'], inplace=True)

  return df


def one_hot_encoder(df, cols):
    """one-hot encoding function for all our categorical columns"""

    for col in cols:
        if("admission" in col):
            dummies = pd.get_dummies(df[col], drop_first=False)
        else:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)   
        df.drop([col],axis=1, inplace=True)
    return df

def preprocessor(df):
  """ preprocessing main function to handle missing values, categorical variables,
      one-hot encoding, nominal values, etc. 
  """
  df.replace("?", np.nan, inplace=True)

  categorical_cols = [col for col in df.columns if df[col].dtype == np.dtype(np.object)]
  df = drop_rows(df, categorical_cols)

  ## one record per patient (where they had max of time_in_hospital)
  df = df.loc[df.groupby("patient_nbr", sort=False)['time_in_hospital'].idxmax()]
  df.drop(columns = ['patient_nbr'], inplace=True)

  #convert categorical 
  df = categorical_to_numerical(df)
  df = map_diagnosis(df)
  df = map_admissions(df)

  #one-hot encoding 
  categorical_columns = [col for col in df.columns if df[col].dtype == np.dtype(object)]
  df = one_hot_encoder(df, categorical_columns)
  df.columns = map(str.lower, df.columns)


  #train-test-split
  target_variable = 'readmitted'
  Y_feature = df[target_variable]
  X_features = df.drop(columns=[target_variable])
  X_train, X_test, y_train, y_test = train_test_split(X_features,Y_feature, test_size=0.2, random_state = 42)


  # normalize of numerical columns
  mm_scaler = MinMaxScaler()
  X_train = pd.DataFrame(mm_scaler.fit_transform(X_train), columns = X_train.columns) 
  X_test = pd.DataFrame(mm_scaler.fit_transform(X_test), columns = X_test.columns)

  return (X_train, X_test, y_train, y_test)