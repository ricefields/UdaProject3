import pandas as pd
import numpy as np
import os
import tensorflow as tf
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3

def reduce_dimension_ndc(df, ndc_code_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    #new_df = df.merge(ndc_code_df[['NDC_Code', 'Non-proprietary Name']], 
    #                  how="inner", left_on='ndc_code', right_on="NDC_Code")
    new_df = pd.merge(df, ndc_code_df[['Non-proprietary Name', 'NDC_Code']],how="left", 
                      left_on='ndc_code', right_on='NDC_Code')
    new_df.drop(columns=["NDC_Code"], inplace=True)
    new_df.rename(columns={"Non-proprietary Name": "generic_drug_name"}, inplace=True)
    new_df.nunique()
    reduce_dim_df = new_df
    return reduce_dim_df

#Question 4
def select_first_encounter (reduce_dim_df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    df = reduce_dim_df.sort_values('encounter_id')
    first_encounter_dataframe = df.groupby(['patient_nbr']).head(1)
    return first_encounter_dataframe



#Question 6
#def patient_dataset_splitter(df, patient_key='patient_nbr'):

def patient_dataset_splitter (df, colname):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    train_dataset     = df.sample(frac=0.6,random_state=0)
    remaining_dataset = df.drop(train_dataset.index)
    valid_dataset     = remaining_dataset.sample(frac=0.5, random_state=0)
    test_dataset      = remaining_dataset.drop(valid_dataset.index)    
    return train_dataset, valid_dataset, test_dataset
    return train, validation, test

#Question 7
def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    #def create_tf_categorical_feature_cols (cat_col_list):
    #one_hot_feature_list = []
    #for i in range (0, len(cat_col_list)):                
     #   vocab = tf.feature_column.categorical_column_with_vocabulary_file(key=cat_col_list[i], 
     #                                                                    vocabulary_file = vocab_file_list[i], 
     #                                                                    num_oov_buckets=1)
     #   one_hot_feature = tf.feature_column.indicator_column(vocab)
     #   one_hot_feature_list.append(one_hot_feature)
    #return one_hot_feature_list 

    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        feature_column = tf.feature_column.categorical_column_with_vocabulary_file(key=c, 
                                                                          vocabulary_file = vocab_file_path)
        print (c, vocab_file_path)
        tf_categorical_feature_column = tf.feature_column.indicator_column(feature_column)
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std
    
#def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
def create_tf_numeric_feature (colname, mean_val, stdev_val, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=mean_val, std=stdev_val)
    tf_numeric_feature = tf.feature_column.numeric_column(key=colname, default_value = 0, 
                                            normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature    

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    # Trial and error based on (1) the output of prob_output_df.describe() above and (2) observed performance below
    student_binary_prediction = df[col].apply(lambda x:1 if x>=6 else 0)
    return student_binary_prediction
