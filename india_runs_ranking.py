"""
India Runs AI Data Challenge — Candidate Ranking Pipeline
=========================================================
Cleaned and fixed version of the Colab notebook.

Fixes applied:
  1. Removed duplicate / redundant re-run cells (data was re-loaded and
     re-scored 3-4 times inside the same notebook session).
  2. final_ranking_score is now returned on the same 0-1 scale as the other
     scores (removed the x10 scaling that inflated values to ~5.7).
  3. Hardcoded /content/ Colab paths replaced with configurable constants
     that default to the local directory.
  4. get_candidate_text_for_embedding now safely handles missing keys in
     career_history, skills and education using .get() guards.
  5. semantic_score values (numpy float32) are cast to Python float before
     JSON serialisation to avoid TypeError.
  6. Unfilled placeholder "YOUR_NOTEBOOK_NAME.ipynb" removed.
  7. years_of_experience is now validated as numeric (falls back to 0).
"""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Detect device (GPU preferred, CPU fallback) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    print(f"GPU detected: {torch.cuda.get_device_name(0)} — running on GPU.")
else:
    print("No CUDA GPU found — running on CPU.")

# --- 0. Configurable paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_FILE = os.path.join(BASE_DIR, "candidates.jsonl")
CANDIDATE_SCHEMA_FILE = os.path.join(BASE_DIR, "candidate_schema.json")
SAMPLE_CANDIDATES_OUTPUT_JSON = os.path.join(BASE_DIR, "sample_candidates.json")
SAMPLE_CANDIDATES_OUTPUT_CSV = os.path.join(BASE_DIR, "sample_candidates.csv")

# --- 1. Load Data ---
print(f"\nLoading {CANDIDATES_FILE}...")
if not os.path.exists(CANDIDATES_FILE):
    raise FileNotFoundError(
        f"{CANDIDATES_FILE} not found. "
        "Please place candidates.jsonl in the same directory as this script."
    )

candidates_data = []
with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            candidates_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping malformed JSON on line {i + 1}: {e}")

df_candidates = pd.DataFrame(candidates_data)
print(f"Successfully loaded {len(df_candidates)} valid candidates.")
print("\nColumns:", df_candidates.columns.tolist())

# Load schema (optional -- just for inspection)
if os.path.exists(CANDIDATE_SCHEMA_FILE):
    with open(CANDIDATE_SCHEMA_FILE, "r", encoding="utf-8") as f:
        candidate_schema = json.load(f)
    print("\nCandidate schema loaded.")
else:
    candidate_schema = {}
    print(f"Warning: {CANDIDATE_SCHEMA_FILE} not found -- proceeding without schema.")

# --- 2. Define Job Description ---
job_description_text = """
We are looking for a highly skilled Software Engineer with expertise in Python,
Machine Learning, and Cloud platforms (AWS/Azure/GCP). The ideal candidate will
have 5+ years of experience, a strong background in data structures and
algorithms, and experience in building scalable backend systems. Experience with
natural language processing (NLP) or large language models (LLMs) is a
significant plus.
"""
print("\nJob Description defined.")

# --- 3. Load Embedding Model ---
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"\nInitialising embedding model: {MODEL_NAME} on {DEVICE.upper()}...")
embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print(f"Embedding model loaded on {DEVICE.upper()}.")

# --- 4. Preprocess Candidate Text for Embeddings ---

def get_candidate_text_for_embedding(row):
    """
    Combine profile summary, career history descriptions, skill names, and
    education details into a single string for semantic embedding.

    FIX: All field accesses now use .get() to safely handle missing keys.
    """
    profile = row.get("profile") or {}
    profile_summary = profile.get("summary", "") if isinstance(profile, dict) else ""

    career_history = row.get("career_history") or []
    career_history_desc = " ".join(
        job.get("description", "")
        for job in career_history
        if isinstance(job, dict)
    )

    skills = row.get("skills") or []
    skills_names = " ".join(
        skill.get("name", "")
        for skill in skills
        if isinstance(skill, dict)
    )

    education = row.get("education") or []
    education_details = " ".join(
        f"{edu.get('degree', '')} in {edu.get('field_of_study', '')}"
        for edu in education
        if isinstance(edu, dict) and (edu.get("degree") or edu.get("field_of_study"))
    )

    combined = f"{profile_summary} {career_history_desc} {skills_names} {education_details}"
    return combined.strip()


print("\nExtracting candidate text for embedding...")
df_candidates["combined_text_for_embedding"] = df_candidates.apply(
    get_candidate_text_for_embedding, axis=1
)
print(f"Done -- {len(df_candidates)} candidates processed.")

# --- 5. Generate Embeddings & Compute Semantic Similarity ---
print("\nGenerating JD embedding...")
jd_embedding = embedding_model.encode(job_description_text, convert_to_numpy=True)
print("JD embedding generated.")

print("\nGenerating candidate embeddings (this may take a few minutes)...")
# Use a larger batch size on GPU for faster throughput
batch_size = 512 if DEVICE == "cuda" else 256
candidate_embeddings = embedding_model.encode(
    df_candidates["combined_text_for_embedding"].tolist(),
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True,
    device=DEVICE,
)
print("Candidate embeddings generated.")

print("\nCalculating cosine similarity scores...")
jd_embedding_reshaped = jd_embedding.reshape(1, -1)
semantic_scores = cosine_similarity(jd_embedding_reshaped, candidate_embeddings)[0]
df_candidates["semantic_score"] = semantic_scores
print("Semantic scores added to DataFrame.")

print("\nTop 5 candidates by semantic similarity:")
print(
    df_candidates[["candidate_id", "semantic_score"]]
    .sort_values("semantic_score", ascending=False)
    .head()
    .to_string()
)

# --- 6. Extract Years of Experience ---

def safe_years_of_experience(profile):
    """
    FIX: Validates that years_of_experience is numeric; falls back to 0.
    """
    if not isinstance(profile, dict):
        return 0.0
    val = profile.get("years_of_experience", 0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


df_candidates["years_of_experience"] = df_candidates["profile"].apply(
    safe_years_of_experience
)
print("\nYears of experience extracted.")

# --- 7. Extract Required Skills from JD ---
SKILL_KEYWORDS = [
    "Python", "Machine Learning", "Cloud", "AWS", "Azure", "GCP",
    "Data Structures", "Algorithms", "Backend Systems", "NLP", "LLMs",
]


def extract_required_skills_from_jd(jd_text):
    """Return lowercase skill keywords that appear in the JD."""
    found = []
    for kw in SKILL_KEYWORDS:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, jd_text, re.IGNORECASE):
            found.append(kw.lower())
    return list(set(found))


required_skills = extract_required_skills_from_jd(job_description_text)
print(f"\nRequired skills from JD: {required_skills}")

# --- 8. Calculate Skill Match Score ---

def calculate_skill_match_score(candidate_skills, req_skills):
    if not req_skills:
        return 0.0
    candidate_skill_names = {
        skill["name"].lower()
        for skill in candidate_skills
        if isinstance(skill, dict) and "name" in skill
    }
    matched = candidate_skill_names.intersection(req_skills)
    return len(matched) / len(req_skills)


print("\nCalculating skill match scores...")
df_candidates["skill_match_score"] = df_candidates["skills"].apply(
    lambda skills: calculate_skill_match_score(
        skills if isinstance(skills, list) else [], required_skills
    )
)
print("Skill match scores calculated.")

# --- 9. Combined Ranking Score ---

def custom_ranking_score(row, jd_min_experience=5.0, exp_boost_threshold=10.0):
    """
    FIX: Returns a score on the 0-1 scale (removed the x10 multiplier that
    inflated values to ~5-6 and made the metric hard to interpret).

    Scoring formula:
      base = 0.6 * semantic_score + 0.4 * skill_match_score
      then apply an experience multiplier.
    """
    semantic_score = float(row["semantic_score"])  # cast from numpy float32
    years_exp = float(row["years_of_experience"])
    skill_match = float(row["skill_match_score"])

    score = 0.6 * semantic_score + 0.4 * skill_match

    if years_exp >= exp_boost_threshold:
        # +5% per year over threshold
        score *= 1 + (years_exp - exp_boost_threshold + 1) * 0.05
    elif years_exp >= jd_min_experience:
        # +2% per year for meeting JD minimum
        score *= 1 + years_exp * 0.02
    elif skill_match == 1.0:
        # Perfect skill match even below minimum experience -> small boost
        score *= 1.05
    else:
        # Penalty for insufficient experience
        score *= 1 - (jd_min_experience - years_exp) * 0.05

    return max(0.0, score)


print("\nApplying custom ranking logic...")
df_candidates["final_ranking_score"] = df_candidates.apply(
    custom_ranking_score, axis=1
)
print("Custom ranking scores calculated.")

print("\nTop 10 candidates after custom ranking:")
print(
    df_candidates[
        [
            "candidate_id",
            "years_of_experience",
            "skill_match_score",
            "semantic_score",
            "final_ranking_score",
        ]
    ]
    .sort_values("final_ranking_score", ascending=False)
    .head(10)
    .to_string()
)

# --- 10. Generate Shortlist and Save Outputs ---
SHORTLIST_COUNT = 50

final_shortlist_df = (
    df_candidates.sort_values("final_ranking_score", ascending=False)
    .head(SHORTLIST_COUNT)
)

# Save JSON
output_candidates = []
for _, row in final_shortlist_df.iterrows():
    profile = row["profile"] if isinstance(row["profile"], dict) else {}
    output_candidates.append(
        {
            "candidate_id": row["candidate_id"],
            # FIX: cast numpy float32 -> Python float to avoid JSON TypeError
            "final_ranking_score": float(row["final_ranking_score"]),
            "years_of_experience": float(row["years_of_experience"]),
            "skill_match_score": float(row["skill_match_score"]),
            "semantic_score": float(row["semantic_score"]),
            "profile_headline": profile.get("headline", ""),
            "profile_summary": profile.get("summary", ""),
        }
    )

print(f"\nSaving top {SHORTLIST_COUNT} candidates -> {SAMPLE_CANDIDATES_OUTPUT_JSON}...")
with open(SAMPLE_CANDIDATES_OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output_candidates, f, indent=4, ensure_ascii=False)
print("JSON saved successfully.")

# Save CSV (top 100)
CSV_COUNT = 100
csv_shortlist_df = (
    df_candidates.sort_values("final_ranking_score", ascending=False)
    .head(CSV_COUNT)
    .copy()
)
csv_shortlist_df["profile_headline"] = csv_shortlist_df["profile"].apply(
    lambda x: x.get("headline", "") if isinstance(x, dict) else ""
)
csv_shortlist_df["profile_summary"] = csv_shortlist_df["profile"].apply(
    lambda x: x.get("summary", "") if isinstance(x, dict) else ""
)
# Cast float32 columns for clean CSV output
for col in ["semantic_score", "final_ranking_score", "skill_match_score"]:
    csv_shortlist_df[col] = csv_shortlist_df[col].astype(float)

output_csv_df = csv_shortlist_df[
    [
        "candidate_id",
        "final_ranking_score",
        "years_of_experience",
        "skill_match_score",
        "semantic_score",
        "profile_headline",
        "profile_summary",
    ]
]
print(f"\nSaving top {CSV_COUNT} candidates -> {SAMPLE_CANDIDATES_OUTPUT_CSV}...")
output_csv_df.to_csv(SAMPLE_CANDIDATES_OUTPUT_CSV, index=False)
print("CSV saved successfully.")

print("\n=== Pipeline complete ===")
print(f"   JSON shortlist : {SAMPLE_CANDIDATES_OUTPUT_JSON}")
print(f"   CSV  shortlist : {SAMPLE_CANDIDATES_OUTPUT_CSV}")

 