import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import pandas as pd

from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser()

#parser.add_argument('-m', "--model",)    
#parser.add_argument('-d', "--data",)    

parser.add_argument('-t', "--task",)    
args = parser.parse_args()


taskID = int(args.task)

Models=["skip", "all-MiniLM-L6-v2", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-mpnet-base-v2", "bert_CLS", "bert_CLS"]

LLMs=["skip", "4o_newnoprompt", "llama_newnoprompt", "4o_newnoprompt", "llama_newnoprompt", "4o_newnoprompt", "llama_newnoprompt"]

#sbert_model = "all-MiniLM-L6-v2"
#dataInput = "4o_all"

#sbert_model = args.model
#dataInput = args.data

sbert_model = Models[taskID]
dataInput = LLMs[taskID]

if sbert_model == "bert_CLS":
    bert_version = 'bert-base-uncased'
    #'mnaylor/psychbert-finetuned-mentalhealth'
    tokenizer = BertTokenizer.from_pretrained(bert_version)
    model = BertModel.from_pretrained(bert_version)
else:
    model = SentenceTransformer(sbert_model)
#model = SentenceTransformer("all-mpnet-base-v2")

#results_df = pd.read_csv(f"~/data/psychagents/experiment_results_{dataInput}.csv")
results_df = pd.read_csv(f"./Data/experiment_results_{dataInput}.csv")

# Load BERT model and tokenizer
device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = model.eval()
model = model.to(device)

# Function to get BERT embeddings
def get_bert_embedding(text):
    inputs = tokenizer(text,padding='max_length', return_tensors = 'pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs[1][0].cpu()
    
# Calculate similarity for each question and condition
def calculate_similarity_bert(results_df):
    similarity_results = []

    for index, row in results_df.iterrows():
        for question in ['Text_SubjectiveLit', 'Text_Anxiety', 'Text_Numeracy', 'Text_TrustPhys']:
            generated_response = row[f"{question} Generated"]
            true_label = row[f"{question} True"]

            if generated_response == "skip":
                continue

            documents = [generated_response, true_label]

            #print(generated_response)
            #print(true_label)

            sentences = [
                generated_response,
                true_label
            ]

            try:
                if sbert_model == "bert_CLS":
                    generated_embedding = get_bert_embedding(generated_response)
                    true_embedding = get_bert_embedding(true_label)
                    similarity = 1 - cosine(generated_embedding, true_embedding)
                else:
                    embeddings = model.encode(sentences)
                    similarity = model.similarity(embeddings, embeddings)
                    similarity = similarity[0][1].detach().item()
            except:
                continue
            
            similarity_results.append({
                'Sample Index': row['Sample Index'],
                'Condition': row['Condition'],
                'Question': question,
                'Similarity': similarity,
                'Generated Response': generated_response,
                'True Label': true_label
            })

        if index % 1000 == 0:
            print(index)
    return pd.DataFrame(similarity_results)

# Perform similarity evaluation
similarity_df_bert_orig = calculate_similarity_bert(results_df)
similarity_df_bert_orig.to_csv(f"similarity_calculations_{dataInput}_{sbert_model}.csv")
