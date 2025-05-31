''' 
    experiment_data_lists = {
        'Demographic only': demographic_list,
        'Demographic + Behavioral': demographic_list + behavioral_list,
        'Demographic + Behavioral + Psychological': demographic_list + behavioral_list + psychological_list,
        'Behavioral only': behavioral_list,
        'Behavioral + Psychological': behavioral_list + psychological_list,
        'Psychological only': psychological_list
    }
'''

EXPERIMENT_TYPE = 'Demographic + Behavioral + Psychological'

# Standard library imports
import os
import json
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise
from umap import UMAP
from openai import OpenAI


from codes.utils import *

file_path = 'Data\\Data_SurveyPlusDemographics.txt'
data = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1')


# Data preprocess
data, psychological_list = convert_to_natural_language(data)
data, fipi_columns = split_fipi_responses(data)
data.drop(columns=['FIPI_response'], inplace=True, errors='ignore')
psychological_list.extend(fipi_columns)


train_data = select_experiment_data(data, EXPERIMENT_TYPE)
train_data = process_data_based_on_experiment(train_data, EXPERIMENT_TYPE)

# add labels column
label_columns = ['Text_SubjectiveLit', 'Text_Anxiety', 'Text_Numeracy', 'Text_TrustPhys']
train_data = pd.concat([train_data, data[label_columns]], axis=1)


train_data = train_data.dropna()
# sample number
sample_data = train_data
sample_data = sample_data.dropna()
#sample_data = train_data.sample(n=30, random_state=42)
sample_data = train_data.sample(n=len(train_data), random_state=42)

labels = sample_data[['Text_SubjectiveLit', 'Text_Anxiety', 'Text_Numeracy', 'Text_TrustPhys']].copy()
labels.reset_index(drop=True, inplace=True)

def convert_to_jsonl_format(dataset, labels, output_dir, condition):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Update categorical columns to include 'no reply' as a category if not present
    for col, content in dataset.items():
        if pd.api.types.is_categorical_dtype(content):
            if 'no reply' not in content.cat.categories:
                dataset[col] = content.cat.add_categories('no reply')

    #print(dataset)
    
    dataset.fillna('no reply', inplace=True)

    if condition == "all_4":
        jsonl_data = []
        for index, row in dataset.iterrows():
            system_prompt = "You should simulate a specified person's persona based on the background information I provided. You are currently visiting a psychologist."
            if 'Demographic' in EXPERIMENT_TYPE:
                system_prompt += convert_to_prompt_demo(row)
            if 'Behavioral' in EXPERIMENT_TYPE:
                system_prompt += convert_to_prompt_behavioral(row)
            if 'Psychological' in EXPERIMENT_TYPE:
                system_prompt += convert_to_prompt_psychological(row)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": Text_SubjectiveLit},
                {"role": "assistant", "content": row['Text_SubjectiveLit']},
                {"role": "user", "content": Text_TrustPhys},
                {"role": "assistant", "content": row['Text_TrustPhys']},
                {"role": "user", "content": Text_Anxiety},
                {"role": "assistant", "content": row['Text_Anxiety']},
                {"role": "user", "content": Text_Numeracy},
                {"role": "assistant", "content": row['Text_Numeracy']}
            ]
            jsonl_data.append({"messages": messages})

        # Save the jsonl file
        filename = os.path.join(output_dir, "all_4.jsonl")
        with open(filename, 'w') as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {filename}")
        
    elif condition == "given_3_hold_1":
        questions = [
            ("Text_SubjectiveLit", Text_SubjectiveLit),
            ("Text_TrustPhys", Text_TrustPhys),
            ("Text_Anxiety", Text_Anxiety),
            ("Text_Numeracy", Text_Numeracy)
        ]
        
        for i in range(4):
            jsonl_data = []
            for index, row in dataset.iterrows():
                system_prompt = "You should simulate a specified person's persona based on the background information I provided. You are currently visiting a psychologist."
                if 'Demographic' in EXPERIMENT_TYPE:
                    system_prompt += convert_to_prompt_demo(row)
                if 'Behavioral' in EXPERIMENT_TYPE:
                    system_prompt += convert_to_prompt_behavioral(row)
                if 'Psychological' in EXPERIMENT_TYPE:
                    system_prompt += convert_to_prompt_psychological(row)

                messages = [{"role": "system", "content": system_prompt}]
                for j, (question, prompt) in enumerate(questions):
                    if i != j:  # Skip the question that we want to hold out
                        messages.append({"role": "user", "content": prompt})
                        messages.append({"role": "assistant", "content": row[question]})
                
                jsonl_data.append({"messages": messages})

            holdout_filename = questions[i][0]  # Use the question name as part of the filename
            filename = os.path.join(output_dir, f"holdout_{holdout_filename}.jsonl")
            with open(filename, 'w') as f:
                for entry in jsonl_data:
                    f.write(json.dumps(entry) + "\n")
            print(f"Saved {filename}")

    elif condition == "conditioning_on_all":
        jsonl_data = []
        for index, row in dataset.iterrows():
            system_prompt = "You should simulate a specified person's persona based on the background information I provided. You are currently visiting a psychologist."
            if 'Demographic' in EXPERIMENT_TYPE:
                system_prompt += convert_to_prompt_demo(row)
            if 'Behavioral' in EXPERIMENT_TYPE:
                system_prompt += convert_to_prompt_behavioral(row)
            if 'Psychological' in EXPERIMENT_TYPE:
                system_prompt += convert_to_prompt_psychological(row)

            messages = [{"role": "system", "content": system_prompt}]
            
            jsonl_data.append({"messages": messages})

        # Save the jsonl file
        filename = os.path.join(output_dir, "conditioning_on_all.jsonl")
        with open(filename, 'w') as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {filename}")

    elif condition == "no_system_prompt":
        jsonl_data = []
        for index, row in dataset.iterrows():
            messages = [
                {"role": "system", "content": 'You are a patient currently visiting a psychologist.'},
                {"role": "user", "content": Text_SubjectiveLit},
                {"role": "assistant", "content": row['Text_SubjectiveLit']},
                {"role": "user", "content": Text_TrustPhys},
                {"role": "assistant", "content": row['Text_TrustPhys']},
                {"role": "user", "content": Text_Anxiety},
                {"role": "assistant", "content": row['Text_Anxiety']},
                {"role": "user", "content": Text_Numeracy},
                {"role": "assistant", "content": row['Text_Numeracy']}
            ]
            jsonl_data.append({"messages": messages})

        # Save the jsonl file
        filename = os.path.join(output_dir, "no_system_prompt.jsonl")
        with open(filename, 'w') as f:
            for entry in jsonl_data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {filename}")

    else:
        raise ValueError("Invalid condition specified.")

    # Save the labels dataset as a CSV file
    labels_filename = os.path.join(output_dir, "labels.csv")
    labels.to_csv(labels_filename, index=False)
    print(f"Saved {labels_filename}")


# 定义输出目录
output_dir = "json_datasets"  


convert_to_jsonl_format(sample_data, labels, output_dir, condition="all_4")
convert_to_jsonl_format(sample_data, labels, output_dir, condition="given_3_hold_1")
convert_to_jsonl_format(sample_data, labels, output_dir, condition="conditioning_on_all")
convert_to_jsonl_format(sample_data, labels, output_dir, condition="no_system_prompt")

