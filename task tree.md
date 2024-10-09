## Task Tree

## Group Simulation
### Some Questions

1. Height recorded as avg 31?

### Prompt design 
   - **Unified System Prompt**: Establish a common **system prompt** that describes the overall characteristics of the group. For example:
     ```text
     You are a professional assistant helping female engineers. This group is typically logical, detail-oriented, and faces career challenges related to work-life balance.
     ```
   - **Personalized User Prompts**: Provide individualized **user prompts** with unique personal details. For example:
     ```text
     I am a 35-year-old female engineer, introverted, and often face challenges in social interactions at work.
     ```
   - **Training**: Combine the system prompt (group-level traits) and user prompts (individual-level traits) during fine-tuning to help the model learn both the shared characteristics and individual differences.



### Testing Progress

run anova to see significance

eval-- llama embedding
scoring on other trained agents

1. Demographic only (finished)
2. Deographic + behavioral (finished)
3. Demographic + behavioral + psychological (finished)
4. Behavioral only (finished)
5. Behavioral + psychological
6. Psychological only

can generative ai improve social science?

### Evaluation methods

1. Embedding similarity (finished)
2. LLM as judge (researching for improvement-- constantly select original model)
3. Other methods (BLEU, ROUGE, METEOR) (On progress)




## Individual Simulation
 Test the agents with different types of input data (e.g., straightforward factual questions vs. complex scenario-based queries) to see how well they handle varying information complexities
demo + beha+ psych
testing on the same people(200 random datapoints)-- comparing generated text and real text
1. all 4 prompts
2. conditioning on questions 
- all 4 
- given3, hold 1

2 baselines
- no system prompt
- no promt + 0shot (200 times)


eval
dist bert embedding (pairwise comparison similarity )-- 
top 5 
llama embedding
few dimensions:
Coherence? 
Sentimental consistancy?  -- passed
Diversity? 
Relevance? 
Human likeness? 
Bias and fairness?
for instance, toxiicity uses Perspective API \
![alt text](image.png)

face validity

'Please read and compare this two paragraphs about the patient's attitude on {PROBLEM}. Please do these tasks:
1. What attitude did they display on each paragraph on a scale of "extremely disagree", "moderately disagree", "neutral", "moderately agree", "extremely agree"？”
2. Rate the difference between two paragraph's attitudes range from 0 to 10. 0 means completely different (for instance, one is "extremely disagree" while the other is "extremely agree"),  2 means highly different (for instance, one is "moderately disagree" while the other is "extremely agree"), and so on, finally 10 means completely same (for instance, one is "extremely agree" while the other is also "extremely agree")
'
