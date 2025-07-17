import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tab-delimited CSV file
csv_path = "fig_data.tsv"  # Adjust the filename as needed
try:
    df = pd.read_csv(csv_path, delimiter="\t")
    print(f"Loaded data from '{csv_path}'")
except FileNotFoundError:
    print(f"Error: '{csv_path}' not found. Please ensure the file exists.")
    exit(1)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Verify required column
required_column = "premise"
if required_column not in df.columns:
    print(f"Error: CSV must contain '{required_column}' column.")
    exit(1)

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# Get emotion labels from model config
emotion_labels = list(model.config.id2label.values())

# Function to compute emotion scores
def get_emotion_scores(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1).squeeze().tolist()
        return probs
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
        return [0.0] * len(emotion_labels)

# Apply emotion scoring to the "premise" column
try:
    emotion_scores = df["premise"].apply(get_emotion_scores)
    emotion_df = pd.DataFrame(emotion_scores.tolist(), columns=emotion_labels)
    
    # Combine original data with emotion scores
    df_full = pd.concat([df, emotion_df], axis=1)
    
    # Save final output
    output_path = "fig_with_emotion_scores1.csv"
    df_full.to_csv(output_path, index=False)
    print(f"Emotion analysis complete. Results saved to '{output_path}'")
    
except Exception as e:
    print(f"Error during emotion analysis or saving: {e}")
    exit(1)
