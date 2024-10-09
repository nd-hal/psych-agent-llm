File				This is the test file name/number where this particular instance appears as a test case in the five-fold CV split
Row				Row no.

********************************************************************************************************************************************************************************************
Q1				How often do you have someone help you read hospital materials?
Q2				How often do you have problems learning about your medical condition because of difficulty understanding written information?
Q3				How often do you have a problem understanding what is told to you about your medical condition?
Q4				How confident are you filling out medical forms by yourself?
FQ1				How often do you need someone to help you when you are given information to read by your doctor, nurse or pharmacist?
FQ2				When you need help, can you easily get hold of someone to assist you?
FQ3				Do you need help to fill in official documents?
ComQ1				When you talk to a doctor or nurse, do you give them all the information they need to help you?
ComQ2				When you talk to a doctor or nurse, do you ask the questions you need to ask?
ComQ3				When you talk to a doctor or nurse, do you make sure they explain anything that you do not understand?
Label_SubjectiveLit		Score on a 0-1 range with 0 being the lowest and 1 being the highest. Formula used was ((Q1+Q2+Q3+Q4)/20+(FQ1+FQ2+FQ3)/6+((4-COMQ1)+(4-COMQ2)+(4-COMQ3))/12)/3
Text_SubjectiveLit		Regarding all the questions you just answered, to what degree do you feel you have the capacity to obtain, process, and understand basic health information and services needed to make appropriate health decisions? Please explain your answer in a few sentences.
********************************************************************************************************************************************************************************************

Demographic_and_behavioral.ipynb
********************************************************************************************************************************************************************************************
BMC1_1				Patient Trust in a Physician - Sometimes your doctor cares more about what is convenient for (him/her) than about your medical needs.
BMC1_2				Patient Trust in a Physician - Your doctor is extremely thorough and careful.
BMC1_3				Patient Trust in a Physician - You completely trust your doctor's decisions about which medical treatments are best for you.
BMC1_4				Patient Trust in a Physician - Your doctor is totally honest in telling you about all of the different treatment options available for your condition.
BMC1_5				Patient Trust in a Physician - All in all, you have complete trust in your doctor.
Label_TrustPhys			Score on a 0-1 range with 0 being the lowest and 1 being the highest. Formula used was ((6-BMC1_1)+(BMC1_2+BMC1_3+BMC1_4+BMC1_5))/25
Text_TrustPhys			In a few sentences, please explain the reasons why you trust or distrust your primary care physician. If you do not have a primary care physician, please answer in regard to doctors in general.
********************************************************************************************************************************************************************************************

********************************************************************************************************************************************************************************************
Emo1_5				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Anxious
Emo1_6				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Upset
Emo1_7				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Discouraged
Emo1_8				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Fearful
Emo1_9				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Worried
Emo1_10				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Uneasy
Emo1_11				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Dread
Emo1_12				Below are some emotions that might be used to describe how you feel when having a doctor give you a health examination. Please think about each emotion carefully and whether a physician health examination made you feel... - Uncertainty
Label_Anxiety			Score on a 0-1 range with 0 being the lowest and 1 being the highest. Formula used was (EMO1_5+EMO1_8+EMO1_9+EMO1_10+EMO1_11+EMO1_12)/42
Text_Anxiety			In a few sentences, please describe what makes you feel most anxious or worried when visiting the doctor's office.
********************************************************************************************************************************************************************************************

HC_1				Please rate your level of agreement with the following statements. - Living life in the best possible health is very important to me.
HC_2				Please rate your level of agreement with the following statements. - Eating right, exercising, and taking preventive measures will keep me healthy for life.
HC_3				Please rate your level of agreement with the following statements. - My health depends on how well I take care of myself.
HC_4				Please rate your level of agreement with the following statements. - I actively try to prevent disease and illness.
HC_5				Please rate your level of agreement with the following statements. - I do everything I can to stay healthy.
Rx2				Overall, how would you describe your health?
Validity			Please select the number fifty-five from the list below:

********************************************************************************************************************************************************************************************
MN1				James has diabetes. His goal is to have his blood sugar between 80 mg/dL and 150 mg/dL in the morning. Which of the following blood sugar readings is within his goal?
MN2				Nathan has a pain rating of 5 on a pain scale of 1 (no pain) to 10 (worst possible pain). One day later than still has pain but not as much. Now, what pain rating might than give?
MN3				Frank has a test done to look for blockages in the arteries of his heart. The doctor said that a person with a higher percent (%) blockage has a high chance of having a heart attack. Which percent (%) blockage has the highest chance of a heart attack?
MN4				Natasha started taking a new medicine that may cause the side effects listed below. Which side effect is Natasha least likely to have?  Side Effect (Chance of Occurring):
MN5				James starts a new blood pressure medicine. The chance of a serious side effect is 0.5%. If 1000 people take this medicine, about how many would be expected to have a serious side effect?
MN6				The PSA (prostate specific antigen) is a blood test that looks for prostate cancer. The test has false alarms so about 30% of men who have an abnormal test turn out not to have prostate cancer. John has an abnormal test result. What is the chance that John has prostate cancer?
MN7				A study found that a new diabetes medicine led to control of blood sugar in 8% more patients than the old medicine. This difference was statistically significant (p=.05). The likelihood that this finding was due to chance alone is best described as less than:
MN8				A nutrition label is shown below. How many calories did Mary eat if she had 2 cups of food?
GHNT1				Call your doctor if you have a temperature of 100.4�F or greater. The thermometer looks like the following: Do you call the doctor?
GHNT2				If 4 people out of 20 have a chance of getting a cold, what would be the risk of getting a cold?
GHNT3				Suppose that the maximum heart rate for a 60 year old woman is 160 beats per minute and that she is told to exercise at 80% of her maximum heart rate. What is 80% of that woman's maximum heart rate? Please fill in the number of beats per minute:
GHNT4				You ate half the container of carrots. How many grams of carbohydrates did you eat? (Please see the nutrition label below).
GHNT5				Your doctor tells you that you have high cholesterol. He informs you that you have a 10% risk of having a heart attack in the next 5 years. If you start on a cholesterol-lowering drug, you can reduce your risk by 30%. What is your 5-year risk if you take the drug?
GHNT6				A mammogram is used to screen women for breast cancer. False positives are tests that incorrectly show a positive result. 85% of positive mammograms are actually false positives. If 1,000 women receive mammograms, and 200 are told there is an abnormal finding, how many women are likely to actually have breast cancer?
Text_Numeracy			In a few sentences, please describe an experience in your life that demonstrated your knowledge of health or medical issues.
Label_Numeracy			Score on a 0-1 range with 0 being the lowest and 1 being the highest. Formula used was (Number of correct responses to MN1-MN8 + Number of correct responses to GHNT1-GHNT6)/14
********************************************************************************************************************************************************************************************

D1				Age: Options for 18 (1) up to 99 (82)
D2				Sex: Male (1); Female (2)
D3				Race: White (1); Black or African American (2); Asian (3); Native American or American Indian (4); Native Hawaiian or Pacific Islander (5); Multiracial or biracial (6); Other (7); I choose not to answer (8)
D4				Education: 8th grade or less (1); Attended high school (2); High school graduate (3); Some college (4); College graduate (5); Some graduate school (6); Grad / professional degree (7); I choose not to answer (8)
D5				Income: Less than $20,000 (1); $20,000 - $34,999 (2); $35,000 - $54,999 (3); $55,000 - $74,999 (4); $75,000 - $89,999 (5); $90,000 or more (6); Do not know / not sure (7); I choose not to answer (8)
D6				Is English your first language?
Dmed_7				What is your height (in feet and inches)?
Dmed_8				How much do you currently weigh (in pounds)?



DMed_1				Approximately how many different prescription drugs are you currently taking on a regular basis?
DMed_2				Do you have a primary care physician?
DMed_3				In the past two years (24 months), how many times have you been to a primary care physician?
DMed_9				On average, how many hours per week do you engage in physical exercise/activity of any kind (e.g., running, walking, biking, swimming, team or individual sports, aerobics, weight-training, etc.)?
Dmed_10				Please describe the overall healthiness of your eating habits:
Dmed_5				During the past 30 days, on how many occasions did you smoke cigarettes?
Dmed_6				On average, how many alcoholic drinks (e.g., glasses of wine, bottles of beer, cocktails) do you consume per week?
