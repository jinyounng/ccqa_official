import os
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline
from bert_score import BERTScorer
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, util

def process_batch(nli_pipeline, text_pairs, batch_size=16):
    class EntailmentDataset(Dataset):
        def __init__(self, text_pairs):
            self.text_pairs = text_pairs
        def __len__(self):
            return len(self.text_pairs)
        def __getitem__(self, idx):
            return self.text_pairs[idx]

    dataset = EntailmentDataset(text_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_results = []
    for batch in dataloader:
        results = nli_pipeline(list(batch))
        if isinstance(results, dict):
            results = [results]
        all_results.extend(results)

    entailment_scores = []
    for result in all_results:
        if isinstance(result, list):
            prob = next((r['score'] for r in result if r['label'].upper() == 'ENTAILMENT'), 0.0)
        else:
            prob = result['score'] if result['label'].upper() == 'ENTAILMENT' else 0.0
        entailment_scores.append(prob)

    return entailment_scores

def normalize_answer(answer):
    """
    Normalize yes/no answers to true/false
    Extract the first yes/no from CoT reasoning text
    """
    # Convert to lowercase for case-insensitive matching
    answer_lower = answer.lower()
    
    # Check for yes/no in the answer string - find the first occurrence
    if "yes" in answer_lower:
        return "true"
    elif "no" in answer_lower:
        return "false"
    # Fallback for direct true/false values
    elif answer_lower == "true":
        return "true"
    elif answer_lower == "false":
        return "false"
    
    # If no clear answer is found, return the original
    return answer_lower

def main():
    # Update to the StrategyQA path
    input_directory = "/data3/jykim/Projects/CCQA_official/Benchmarks/StrategyQA/Results/ccqa_result"
    device = 0 if torch.cuda.is_available() else -1
    print(f"Initializing NLI model on device {device}...")
    nli_pipeline = pipeline("text-classification", model="roberta-large-mnli", device=device, batch_size=16)

    print("Initializing BERTScorer and SBERT model...")
    scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, idf=False, device=device)
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    json_files = sorted(glob.glob(os.path.join(input_directory, "*.json")))
    print(f"Found {len(json_files)} JSON files")

    # Create output directory
    output_dir = os.path.join(input_directory, "cosine_precision_nli_scored")
    os.makedirs(output_dir, exist_ok=True)

    for json_file in tqdm(json_files, desc="Processing files"):
        filename = os.path.basename(json_file)
        output_file = os.path.join(output_dir, filename)

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'results' not in data:
                tqdm.write(f"Skipping {filename}: 'results' key not found")
                continue

            for result in data['results']:
                original_question = result.get('question', '')
                if not original_question:
                    continue

                # Handle answer normalization (yes/no to true/false conversion)
                if 'answer' in result:
                    result['original_answer'] = result['answer']
                    result['answer'] = normalize_answer(result['answer'])

                if 'generated_answer' in result:
                    result['original_generated_answer'] = result['generated_answer']
                    result['generated_answer'] = normalize_answer(result['generated_answer'])

                # Process regenerated questions
                regenerated_questions = []
                regenerated_question_keys = []
                for j in range(1, 6):
                    key = f'regenerated_question_{j}'
                    if key in result and result[key]:
                        regenerated_questions.append(result[key])
                        regenerated_question_keys.append(j)

                if not regenerated_questions:
                    continue

                P, _, _ = scorer.score([original_question] * len(regenerated_questions), regenerated_questions)
                precision_scores = [p.item() for p in P]

                backward_texts = [f"{original_question} </s> {gen_q}" for gen_q in regenerated_questions]
                backward_entailment = process_batch(nli_pipeline, backward_texts)

                all_sentences = [original_question] + regenerated_questions
                embeddings = sbert_model.encode(all_sentences, convert_to_tensor=True)
                cosine_scores = util.cos_sim(embeddings[0], embeddings[1:]).squeeze().tolist()
                
                # Make sure cosine_scores is a list (even for single item)
                if not isinstance(cosine_scores, list):
                    cosine_scores = [cosine_scores]

                result['similarity_scores'] = {
                    f"question_{regenerated_question_keys[i]}": {
                        "precision": round(precision_scores[i], 4),
                        "entailment_backward": round(backward_entailment[i], 4),
                        "cosine": round(cosine_scores[i], 4)
                    } for i in range(len(regenerated_question_keys))
                }

                scores_with_index = [
                    (regenerated_question_keys[i], precision_scores[i]) for i in range(len(regenerated_question_keys))
                ]
                scores_with_index.sort(key=lambda x: x[1], reverse=True)
                result['most_similar_idxs'] = [idx for idx, _ in scores_with_index]

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            tqdm.write(f"Saved → {output_file}")

        except Exception as e:
            tqdm.write(f"Error processing {json_file}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\nAll processing completed!")

if __name__ == "__main__":
    main()