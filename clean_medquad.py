"""
MedQuAD cleaning and splitting script (v1)
Source: Kaggle pythonafroz/medquad-medical-question-answer-for-ai-research
Steps: rename columns -> drop exact duplicate (Question, Answer) pairs
       -> drop rows with empty answers -> add record_id
       -> split by normalised question (strip + lowercase), 70/15/15, seed 3407
Usage: python clean_medquad.py <path_to_medquad.csv> <output_dir>
"""
import sys, os
import numpy as np
import pandas as pd

SEED = 3407

def main(raw_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = pd.read_csv(raw_path).rename(columns={"question": "Question", "answer": "Answer"})

    dup_mask = raw.duplicated(subset=["Question", "Answer"], keep="first")
    step1 = raw[~dup_mask]
    empty_mask = step1.Answer.isna() | (step1.Answer.astype(str).str.strip() == "")
    final = step1[~empty_mask].reset_index(drop=True)
    final.insert(0, "record_id", range(len(final)))
    final.to_csv(os.path.join(out_dir, "medquad_final_v1.csv"), index=False)

    final["q_key"] = final.Question.str.strip().str.lower()
    keys = np.array(sorted(final.q_key.unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(keys)
    n = len(keys)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    final["split"] = np.where(final.q_key.isin(train_keys), "train",
                      np.where(final.q_key.isin(val_keys), "val", "test"))
    for s in ["train", "val", "test"]:
        part = final[final.split == s]
        part[["record_id"]].to_csv(os.path.join(out_dir, f"{s}_ids.csv"), index=False)
        part.drop(columns=["q_key", "split"]).to_csv(os.path.join(out_dir, f"{s}.csv"), index=False)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
