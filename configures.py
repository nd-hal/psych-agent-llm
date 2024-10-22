


Text_Numeracy = "In a few sentences, please describe an experience in your life that demonstrated your knowledge of health or medical issues."
Text_Anxiety = "In a few sentences, please describe what makes you feel most anxious or worried when visiting the doctor's office."
Text_TrustPhys = "In a few sentences, please explain the reasons why you trust or distrust your primary care physician. If you do not have a primary care physician, please answer in regard to doctors in general."
Text_SubjectiveLit = "In a few sentences, please describe to what degree do you feel you have the capacity to obtain, process, and understand basic health information and services needed to make appropriate health decisions?"

#"To what degree do you feel you have the capacity to obtain, process, and understand basic health information and services needed to make appropriate health decisions? Please explain your answer in a few sentences."

#BLEU/ROUGE

demographic_list = [
    "D1",
    "D2",
    "D3",
    "D4",
    "D5",
    'D6',
    'Dmed_7',
    'Dmed_8'
]

behavioral_list  = [
    "HC_1",
    "HC_2",
    "HC_3",
    "HC_4",
    "HC_5"
]

psychological_list= ['FIPI_1_response',
 'FIPI_2_response',
 'FIPI_3_response',
 'FIPI_4_response',
 'FIPI_5_response']


text_label_list = [
    "Text_Numeracy",
    "Text_Anxiety",
    "Text_TrustPhys",
    "Text_SubjectiveLit"
]

cat_label_list = [
    "Label_Numeracy",
    "Label_Anxiety",
    "Label_TrustPhys",
    "Label_SubjectiveLit",
    
]


models = {
    'Demographic only': demographic_list,
    'Demographic + Behavioral': demographic_list + behavioral_list,
    'Demographic + Behavioral + Psychological': demographic_list + behavioral_list + psychological_list,
    'Behavioral only': behavioral_list,
    'Behavioral + Psychological': ['ft:gpt-4o-mini-2024-07-18:personal:psych-beha-0:9yBrsJdE',
                                   'ft:gpt-4o-mini-2024-07-18:personal:beha-psych-1:9yCIptNd',
                                   'ft:gpt-4o-mini-2024-07-18:personal:psych-beha-2:9yCQEcGR',
                                   'ft:gpt-4o-mini-2024-07-18:personal:beha-psych-3:9yCbRRyB',
                                   'ft:gpt-4o-mini-2024-07-18:personal:psych-beha-4:9yCq4RIv',
                                   "ft:gpt-4o-mini-2024-07-18:personal:psych-beha-5:9yCr6mW4",
                                   "ft:gpt-4o-mini-2024-07-18:personal:psych-beha-6:9yDE9Qtq",
                                   "ft:gpt-4o-mini-2024-07-18:personal:psych-beha-7:9yDEIRg5",
                                   'gpt-4o-mini'
                                   ],
    'Psychological only': psychological_list
}