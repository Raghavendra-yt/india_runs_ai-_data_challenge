"""
Generates a clean, fixed Jupyter notebook from scratch.
Run this script once to produce: Copy_of_Untitled2_fixed.ipynb
"""

import json

# Each cell is (cell_type, source_lines)
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "source": text})


def code(source_text, cell_id=""):
    cells.append({"cell_type": "code", "source": source_text, "id": cell_id})


# ── Markdown header ──────────────────────────────────────────────────────────
md("""## India Runs AI — Candidate Ranking Pipeline

This notebook loads the provided candidate data, computes semantic similarity
against a job description using a SentenceTransformer model, applies a skill
and experience scoring function, and outputs a ranked shortlist.

**Fixes applied over the original notebook:**
1. Duplicate / redundant re-run cells removed.
2. `final_ranking_score` is now on the 0–1 scale (original multiplied by 10).
3. Hardcoded `/content/` Colab paths replaced with flexible `BASE_DIR` constant.
4. `get_candidate_text_for_embedding` uses `.get()` guards for all nested fields.
5. `numpy float32` values cast to `float` before JSON serialisation.
6. Unfilled placeholder `"YOUR_NOTEBOOK_NAME.ipynb"` removed.
7. `years_of_experience` validated as numeric (falls back to 0).
""")

# ── Cell 1: Install libraries ────────────────────────────────────────────────
md("## Setup: Install Libraries")

code("""\
# Install required libraries
!pip install pandas sentence-transformers scikit-learn -q
print("Libraries installed successfully.")
""", "install")

# ── Cell 2: Imports & configuration ─────────────────────────────────────────
md("## Configuration and Imports")

code("""\
import json
import os
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Configurable paths ──────────────────────────────────────────────────────
# On Google Colab set BASE_DIR = '/content'; locally use the folder that
# contains candidates.jsonl.
BASE_DIR = '/content'   # <-- change if running locally

CANDIDATES_FILE             = os.path.join(BASE_DIR, 'candidates.jsonl')
CANDIDATE_SCHEMA_FILE       = os.path.join(BASE_DIR, 'candidate_schema.json')
SAMPLE_CANDIDATES_JSON      = os.path.join(BASE_DIR, 'sample_candidates.json')
SAMPLE_CANDIDATES_CSV       = os.path.join(BASE_DIR, 'sample_candidates.csv')

SHORTLIST_COUNT = 50   # top-N saved to JSON
CSV_COUNT       = 100  # top-N saved to CSV

print("Configuration done.")
""", "config")

# ── Cell 3: Load data ────────────────────────────────────────────────────────
md("## Load Candidate Data and Schema")

code("""\
print(f"\\nLoading {CANDIDATES_FILE}...")
assert os.path.exists(CANDIDATES_FILE), (
    f"{CANDIDATES_FILE} not found. Please upload it first."
)

candidates_data = []
with open(CANDIDATES_FILE, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            candidates_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping malformed JSON on line {i+1}: {e}")

df_candidates = pd.DataFrame(candidates_data)
print(f"Successfully loaded {len(df_candidates):,} valid candidates.")
print("Columns:", df_candidates.columns.tolist())
display(df_candidates.head())

# Load schema (informational only)
if os.path.exists(CANDIDATE_SCHEMA_FILE):
    with open(CANDIDATE_SCHEMA_FILE, 'r', encoding='utf-8') as f:
        candidate_schema = json.load(f)
    print("\\nCandidate schema loaded.")
    display(candidate_schema)
else:
    candidate_schema = {}
    print(f"Warning: {CANDIDATE_SCHEMA_FILE} not found.")
""", "load_data")

# ── Cell 4: Job description & model ─────────────────────────────────────────
md("## Define Job Description and Load Embedding Model")

code("""\
# ── Job Description ─────────────────────────────────────────────────────────
job_description_text = \"\"\"
We are looking for a highly skilled Software Engineer with expertise in Python,
Machine Learning, and Cloud platforms (AWS/Azure/GCP). The ideal candidate will
have 5+ years of experience, a strong background in data structures and
algorithms, and experience in building scalable backend systems. Experience with
natural language processing (NLP) or large language models (LLMs) is a
significant plus.
\"\"\"
print("Job Description defined.")

# ── Embedding model ─────────────────────────────────────────────────────────
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
print(f"\\nLoading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")
""", "jd_and_model")

# ── Cell 5: Preprocess candidate text ────────────────────────────────────────
md("## Preprocess Candidate Data for Embeddings")

code("""\
def get_candidate_text_for_embedding(row):
    \"\"\"
    Combine profile summary, career history, skill names and education into
    a single string for semantic embedding.

    FIX: All nested accesses use .get() to handle missing/None values.
    \"\"\"
    profile = row.get('profile') or {}
    summary = profile.get('summary', '') if isinstance(profile, dict) else ''

    career_history = row.get('career_history') or []
    career_text = ' '.join(
        job.get('description', '')
        for job in career_history
        if isinstance(job, dict)
    )

    skills = row.get('skills') or []
    skills_text = ' '.join(
        skill.get('name', '')
        for skill in skills
        if isinstance(skill, dict)
    )

    education = row.get('education') or []
    edu_text = ' '.join(
        f\"{edu.get('degree', '')} in {edu.get('field_of_study', '')}\"
        for edu in education
        if isinstance(edu, dict) and (edu.get('degree') or edu.get('field_of_study'))
    )

    return f\"{summary} {career_text} {skills_text} {edu_text}\".strip()


print("Extracting candidate text for embedding...")
df_candidates['combined_text_for_embedding'] = df_candidates.apply(
    get_candidate_text_for_embedding, axis=1
)
print(f"Done — {len(df_candidates):,} candidates processed.")
print("\\nSample text for first candidate:")
print(df_candidates.loc[0, 'combined_text_for_embedding'][:400] + '...')
""", "preprocess")

# ── Cell 6: Embeddings & similarity ─────────────────────────────────────────
md("## Generate Embeddings and Calculate Semantic Similarity")

code("""\
# JD embedding
print("Generating JD embedding...")
jd_embedding = embedding_model.encode(job_description_text)
print("JD embedding generated.")

# Candidate embeddings
print("\\nGenerating candidate embeddings (may take a few minutes for 100k rows)...")
candidate_embeddings = embedding_model.encode(
    df_candidates['combined_text_for_embedding'].tolist(),
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True,
)
print("Candidate embeddings generated.")

# Cosine similarity
print("\\nCalculating semantic similarity scores...")
semantic_scores = cosine_similarity(jd_embedding.reshape(1, -1), candidate_embeddings)[0]
df_candidates['semantic_score'] = semantic_scores
print("Semantic scores added to DataFrame.")

print("\\nTop 5 candidates by semantic similarity:")
display(
    df_candidates[['candidate_id', 'semantic_score']]
    .sort_values('semantic_score', ascending=False)
    .head()
)
""", "embeddings")

# ── Cell 7: Experience, skills, scoring ──────────────────────────────────────
md("## Advanced Filtering: Experience and Skill Matching")

code("""\
# ── Years of experience ─────────────────────────────────────────────────────
def safe_years_of_experience(profile):
    \"\"\"
    FIX: validates that years_of_experience is numeric; falls back to 0.
    \"\"\"
    if not isinstance(profile, dict):
        return 0.0
    val = profile.get('years_of_experience', 0)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


df_candidates['years_of_experience'] = df_candidates['profile'].apply(
    safe_years_of_experience
)
print("Years of experience extracted.")

# ── Skill extraction from JD ────────────────────────────────────────────────
SKILL_KEYWORDS = [
    'Python', 'Machine Learning', 'Cloud', 'AWS', 'Azure', 'GCP',
    'Data Structures', 'Algorithms', 'Backend Systems', 'NLP', 'LLMs',
]


def extract_required_skills_from_jd(jd_text):
    found = []
    for kw in SKILL_KEYWORDS:
        if re.search(r'\\b' + re.escape(kw) + r'\\b', jd_text, re.IGNORECASE):
            found.append(kw.lower())
    return list(set(found))


required_skills = extract_required_skills_from_jd(job_description_text)
print(f"\\nRequired skills from JD: {required_skills}")

# ── Skill match score ───────────────────────────────────────────────────────
def calculate_skill_match_score(candidate_skills, req_skills):
    if not req_skills:
        return 0.0
    candidate_skill_names = {
        s['name'].lower()
        for s in candidate_skills
        if isinstance(s, dict) and 'name' in s
    }
    return len(candidate_skill_names & set(req_skills)) / len(req_skills)


print("\\nCalculating skill match scores...")
df_candidates['skill_match_score'] = df_candidates['skills'].apply(
    lambda s: calculate_skill_match_score(s if isinstance(s, list) else [], required_skills)
)
print("Skill match scores calculated.")

# ── Combined ranking score ──────────────────────────────────────────────────
def custom_ranking_score(row, jd_min_experience=5.0, exp_boost_threshold=10.0):
    \"\"\"
    FIX: score is on 0-1 scale. The original notebook multiplied by 10
    (making scores read as ~5.7 instead of ~0.57), which was misleading.

    base = 0.6 * semantic_score + 0.4 * skill_match_score
    then boosted/penalised by years of experience.
    \"\"\"
    semantic  = float(row['semantic_score'])   # cast numpy float32
    years_exp = float(row['years_of_experience'])
    skill     = float(row['skill_match_score'])

    score = 0.6 * semantic + 0.4 * skill

    if years_exp >= exp_boost_threshold:
        score *= 1 + (years_exp - exp_boost_threshold + 1) * 0.05
    elif years_exp >= jd_min_experience:
        score *= 1 + years_exp * 0.02
    elif skill == 1.0:
        score *= 1.05
    else:
        score *= 1 - (jd_min_experience - years_exp) * 0.05

    return max(0.0, score)


print("\\nApplying custom ranking logic...")
df_candidates['final_ranking_score'] = df_candidates.apply(custom_ranking_score, axis=1)
print("Ranking scores calculated.")

print("\\nTop 10 candidates after custom ranking:")
display(
    df_candidates[['candidate_id', 'years_of_experience', 'skill_match_score',
                   'semantic_score', 'final_ranking_score']]
    .sort_values('final_ranking_score', ascending=False)
    .head(10)
)
""", "scoring")

# ── Cell 8: Save outputs ──────────────────────────────────────────────────────
md("## Save Ranked Shortlist")

code("""\
# ── Save JSON (top SHORTLIST_COUNT) ────────────────────────────────────────
final_shortlist_df = (
    df_candidates.sort_values('final_ranking_score', ascending=False)
    .head(SHORTLIST_COUNT)
)

output_candidates = []
for _, row in final_shortlist_df.iterrows():
    profile = row['profile'] if isinstance(row['profile'], dict) else {}
    output_candidates.append({
        'candidate_id':        row['candidate_id'],
        # FIX: cast numpy float32 -> Python float to avoid JSON TypeError
        'final_ranking_score': float(row['final_ranking_score']),
        'years_of_experience': float(row['years_of_experience']),
        'skill_match_score':   float(row['skill_match_score']),
        'semantic_score':      float(row['semantic_score']),
        'profile_headline':    profile.get('headline', ''),
        'profile_summary':     profile.get('summary', ''),
    })

print(f"\\nSaving top {SHORTLIST_COUNT} candidates -> {SAMPLE_CANDIDATES_JSON}...")
with open(SAMPLE_CANDIDATES_JSON, 'w', encoding='utf-8') as f:
    json.dump(output_candidates, f, indent=4, ensure_ascii=False)
print("JSON saved.")

# ── Save CSV (top CSV_COUNT) ────────────────────────────────────────────────
csv_df = (
    df_candidates.sort_values('final_ranking_score', ascending=False)
    .head(CSV_COUNT)
    .copy()
)
csv_df['profile_headline'] = csv_df['profile'].apply(
    lambda x: x.get('headline', '') if isinstance(x, dict) else ''
)
csv_df['profile_summary'] = csv_df['profile'].apply(
    lambda x: x.get('summary', '') if isinstance(x, dict) else ''
)
for col in ['semantic_score', 'final_ranking_score', 'skill_match_score']:
    csv_df[col] = csv_df[col].astype(float)

output_csv = csv_df[['candidate_id', 'final_ranking_score', 'years_of_experience',
                      'skill_match_score', 'semantic_score',
                      'profile_headline', 'profile_summary']]

print(f"Saving top {CSV_COUNT} candidates -> {SAMPLE_CANDIDATES_CSV}...")
output_csv.to_csv(SAMPLE_CANDIDATES_CSV, index=False)
print("CSV saved.")

print("\\nFirst 5 rows of CSV:")
display(output_csv.head())

print("\\n=== Pipeline complete ===")
print(f"  JSON: {SAMPLE_CANDIDATES_JSON}")
print(f"  CSV : {SAMPLE_CANDIDATES_CSV}")
""", "save_outputs")

# ── Build notebook dict ──────────────────────────────────────────────────────
def make_nb_cell(cell):
    if cell["cell_type"] == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": cell["source"].splitlines(keepends=True),
        }
    else:
        source_lines = cell["source"].splitlines(keepends=True)
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"id": cell.get("id", "")},
            "outputs": [],
            "source": source_lines,
        }


notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {
            "provenance": [],
            "gpuType": "T4",
        },
        "accelerator": "GPU",
    },
    "cells": [make_nb_cell(c) for c in cells],
}

output_path = "Copy_of_Untitled2_fixed.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Fixed notebook written to: {output_path}")
