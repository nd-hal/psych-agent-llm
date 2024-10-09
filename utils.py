import pandas as pd
import re
from configures import *

# 定义数值到自然语言的映射
def map_values_to_text(column, value):
    mappings = {
        'ATQ': {
            1: "never",
            2: "rarely",
            3: "sometimes",
            4: "often",
            5: "always"
        },
        'TWB': {
            1: "strongly disagree",
            2: "disagree",
            3: "neutral",
            4: "agree",
            5: "strongly agree"
        },
        'RA1': {
            1: "very unlikely",
            2: "unlikely",
            3: "likely",
            4: "very likely",
            5: "extremely likely"
        },
        'STAI': {
            1: "never",
            2: "rarely",
            3: "sometimes",
            4: "often",
            5: "always"
        },
        'BSCS': {
            1: "strongly disagree",
            2: "disagree",
            3: "neutral",
            4: "agree",
            5: "strongly agree"
        },
        'REI': {
            1: "strongly disagree",
            2: "disagree",
            3: "neutral",
            4: "agree",
            5: "strongly agree"
        },
        'FIPI': {
            1: "strongly disagree",
            2: "disagree",
            3: "neutral",
            4: "agree",
            5: "strongly agree"
        }
    }
    prefix = column.split('_')[0]
    return mappings[prefix].get(value, "unknown")

# 为每类问题定义一个关键词
keywords = {
    'ATQ': "negative thoughts",
    'TWB': "well-being",
    'RA1': "risk activities",
    'STAI': "feelings",
    'BSCS': "self-control",
    'REI': "thinking style",
    'FIPI_1': "extraverted and enthusiastic",
    'FIPI_2': "agreeable and kind",
    'FIPI_3': "dependable and organized",
    'FIPI_4': "emotionally stable and calm",
    'FIPI_5': "open to experience and imaginative"
}

# 计算每个大主题下的小问题的众数
def calculate_mode(data):
    modes = {}
    for prefix in set(key.split('_')[0] for key in keywords.keys()):
        relevant_columns = [col for col in data.columns if col.startswith(prefix)]
        if relevant_columns:
            modes[prefix] = data[relevant_columns].mode(axis=1)[0]
    return modes

# 将数值数据转换为自然语言描述并存储在单独的列中
def convert_to_natural_language(data):
    new_columns = []
    modes = calculate_mode(data)
    for prefix, mode_values in modes.items():
        responses = []
        for index in range(len(data)):
            text = map_values_to_text(prefix, mode_values.iloc[index])
            keyword = keywords.get(prefix) or keywords.get(f"{prefix}_{index + 1}")
            # 生成自然语言描述
            if 'ATQ' in prefix:
                responses.append(f"Over the last week, you have had {keyword} {text}.")
            elif 'TWB' in prefix:
                responses.append(f"{text} that {keyword} is good.")
            elif 'RA1' in prefix:
                responses.append(f"You are {text} to engage in {keyword}.")
            elif 'STAI' in prefix:
                responses.append(f"You feel {keyword} {text}.")
            elif 'BSCS' in prefix:
                responses.append(f"You {text} that your {keyword} is strong.")
            elif 'REI' in prefix:
                responses.append(f"You {text} with your {keyword}.")
            elif 'FIPI' in prefix:
                responses.append(f"You {text} that you are {keyword}.")
        col_name = f'{prefix}_response'
        data[col_name] = responses
        new_columns.append(col_name)
    return data, new_columns



# 将FIPI部分拆分成5个单独的列
def split_fipi_responses(data):
    fipi_columns = []
    for i in range(1, 6):
        responses = []
        prefix = f'FIPI_{i}'
        for index in range(len(data)):
            text = map_values_to_text(prefix, data[f'FIPI_{i}'].iloc[index])
            keyword = keywords[prefix]
            responses.append(f"You {text} that you are {keyword}.")
        col_name = f'FIPI_{i}_response'
        data[col_name] = responses
        fipi_columns.append(col_name)
    return data, fipi_columns



def select_experiment_data(data, experiment_type):
    # Dictionary to map experiment type to the required data lists
    experiment_data_lists = {
        'Demographic only': demographic_list,
        'Demographic + Behavioral': demographic_list + behavioral_list,
        'Demographic + Behavioral + Psychological': demographic_list + behavioral_list + psychological_list,
        'Behavioral only': behavioral_list,
        'Behavioral + Psychological': behavioral_list + psychological_list,
        'Psychological only': psychological_list
    }

    # Get the list of columns based on experiment type
    selected_columns = experiment_data_lists.get(experiment_type, [])

    # Select the data from these columns
    train_data = data[selected_columns]

    return train_data



# 定义一个函数来清理数值数据
def clean_numeric(value):
    if isinstance(value, str):
        # 使用正则表达式提取数值部分
        numeric_value = re.findall(r'\d+', value)
        if numeric_value:
            return float(numeric_value[0])
        else:
            return None
    return value


def process_data_based_on_experiment(df, experiment_type):
    if "Demographic" in experiment_type:
        df = process_demographic_data(df)
    if "Behavioral" in experiment_type:
        df = process_behavioral_data(df)
    
    return df

def process_demographic_data(df):
    # Process demographic data if it exists
    demographic_columns = ['D1', 'D4', 'D5']  # Add more columns as needed
    for column in demographic_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_numeric)
    return df

def process_behavioral_data(df):
    # Process behavioral data if it exists
    behavioral_columns = ['HC_1', 'HC_2', 'HC_3', 'HC_4', 'HC_5']  # Add more columns as needed
    for column in behavioral_columns:
        if column in df.columns:
            df[column] = df[column].apply(clean_numeric)
            df = convert_to_categorical(df, column)
    
    return df

def convert_to_categorical(df, column):
    df[column] = pd.cut(df[column], bins=[0, 2, 4, 5], labels=['Disagree', 'Neutral', 'Agree'])
    return df

def convert_to_prompt_demo(row):
    # Age
    age = f"{row['D1'] + 18} years old" if row['D1'] != 'no reply' else "unknown age"
    
    # Sex
    sex = "male" if row['D2'] == 1 else "female" if row['D2'] == 2 else "unknown sex"
    
    # Race
    race_options = [
        "White", "Black or African American", "Asian", 
        "Native American or American Indian", "Native Hawaiian or Pacific Islander", 
        "Multiracial or biracial", "Other", "Prefer not to say"
    ]
    race = race_options[int(row['D3']) - 1] if row['D3'] != 'no reply' else "unknown race"
    
    # Education
    education_options = [
        "8th grade or less", "some high school education", "high school graduate", 
        "some college education", "college graduate", "some graduate school education", 
        "a graduate or professional degree", "prefer not to say"
    ]
    education = education_options[int(row['D4']) - 1] if row['D4'] != 'no reply' else "unknown educational background"
    
    # Income
    income_options = [
        "less than $20,000", "$20,000 - $34,999", "$35,000 - $54,999", 
        "$55,000 - $74,999", "$75,000 - $89,999", "$90,000 or more", 
        "unknown income", "prefer not to say"
    ]
    income = income_options[int(row['D5']) - 1] if row['D5'] != 'no reply' else "unknown income"
    
    # Language
    english_first_lang = "English is your first language" if row['D6'] == 1 else \
                         "English is not your first language" if row['D6'] == 2 else \
                         "it is unknown if English is your first language"
    
    # Physical measurements
    if row['Dmed_7'] != 'no reply':
        total_inches = int(row['Dmed_7'])
        feet = 3 + (total_inches // 12)
        inches = total_inches % 12
        height = f"{feet} feet {inches} inches tall"
    else:
        height = "unknown height"

    weight = f"weighs {row['Dmed_8']} pounds" if row['Dmed_8'] != 'no reply' else "unknown weight"
    
    # Combine all into one prompt
    prompt = (
        f"You are {age}, {sex}, of {race} descent. You have {education} and an annual income of {income}. "
        f"{english_first_lang}. You are {height} and {weight}."
    )
    
    return prompt


def convert_to_prompt_behavioral(row):
    # Initialize the intro
    intro = ""

    # Combine health consciousness responses into a natural language prompt
    if row['HC_1'] == 'Agree':
        intro += "Living in the best possible health is very important to you. "
    elif row['HC_1'] == 'Neutral':
        intro += "You find it moderately important to live in the best possible health. "
    elif row['HC_1'] == 'Disagree':
        intro += "You do not prioritize living in the best possible health. "

    if row['HC_2'] == 'Agree':
        intro += "You believe that eating right, exercising, and taking preventive measures will keep you healthy for life. "
    elif row['HC_2'] == 'Neutral':
        intro += "You think that maintaining a healthy lifestyle may or may not guarantee lifelong health. "
    elif row['HC_2'] == 'Disagree':
        intro += "You do not believe that eating right, exercising, and taking preventive measures are enough to keep you healthy for life. "

    if row['HC_3'] == 'Agree':
        intro += "You think your health depends on how well you take care of yourself. "
    elif row['HC_3'] == 'Neutral':
        intro += "You believe your health somewhat depends on your self-care efforts. "
    elif row['HC_3'] == 'Disagree':
        intro += "You do not think your health is significantly influenced by how well you take care of yourself. "

    if row['HC_4'] == 'Agree':
        intro += "You actively try to prevent disease and illness. "
    elif row['HC_4'] == 'Neutral':
        intro += "You sometimes try to prevent disease and illness, but it's not a priority. "
    elif row['HC_4'] == 'Disagree':
        intro += "You do not actively try to prevent disease and illness. "

    if row['HC_5'] == 'Agree':
        intro += "You do everything you can to stay healthy. "
    elif row['HC_5'] == 'Neutral':
        intro += "You make some efforts to stay healthy, but you don't do everything possible. "
    elif row['HC_5'] == 'Disagree':
        intro += "You do not do everything you can to stay healthy. "

    return intro


def convert_to_prompt_psychological(row):
    intro = "You are someone who "
    # 添加 FIPI 各个部分的 response
    intro += row['FIPI_1_response'] + " "
    intro += row['FIPI_2_response'] + " "
    intro += row['FIPI_3_response'] + " "
    intro += row['FIPI_4_response'] + " "
    intro += row['FIPI_5_response'] + " "

    return intro.strip()
