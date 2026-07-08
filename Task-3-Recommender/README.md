# Project 3: AI Recommendation Logic — Tech Stack Recommender

**DecodeLabs Industrial Training Kit — Batch 2026**

A content-based recommendation engine that maps a user's skills and
interests to the most relevant tech career roles (e.g. Data Scientist,
DevOps Engineer, Frontend Developer), using **TF-IDF vectorization** and
**Cosine Similarity** — the same core techniques behind real-world
recommendation engines like Netflix or Amazon.

## 🧠 How It Works (the pipeline)

1. **Dataset (`raw_skills.csv`)** – 10 tech job roles, each tagged with a
   list of associated skills.
2. **Input** – the user types in at least 3 skills or interests
   (e.g. `Python, Cloud Computing, Automation`).
3. **Vector Mapping (TF-IDF)** – every job role's skill list, plus the
   user's own skill list, is converted into a numeric vector. TF-IDF
   automatically gives more weight to specific/rare skills (like
   "TensorFlow") and less weight to generic ones (like "SQL", which
   appears everywhere).
4. **Scoring (Cosine Similarity)** – the user's vector is compared against
   every job role's vector. The score (0 to 1) measures how closely
   aligned they are, regardless of how many skills either list has.
5. **Sorting & Filtering** – roles are ranked highest-score-first, and only
   the **Top 3** are shown, to avoid overwhelming the user.

## 🚀 How to Run

```bash
pip install -r requirements.txt
python recommender.py
```

Then type in your skills when prompted:
Your skills: Python, Cloud Computing, Automation
TOP RECOMMENDATIONS FOR YOU

Cloud Architect   — 35.7% match
DevOps Engineer   — 32.2% match
Data Scientist    — 9.8% match


## 🧪 Example Test Cases

| Input Skills | Top Result |
|---|---|
| Python, Cloud Computing, Automation | Cloud Architect |
| Figma, UI Design, Prototyping | UI/UX Designer |
| Python, Machine Learning, Statistics | Machine Learning Engineer |

## 🛠️ Skills Demonstrated

- Content-based filtering (as opposed to collaborative filtering)
- Feature extraction / vectorization with TF-IDF
- Similarity scoring with Cosine Similarity
- The full Input → Process → Output recommendation pipeline
- Ranking and filtering (Top-N results)

## 📌 Notes / Possible Extensions

- Add more job roles or a bigger skill vocabulary to `raw_skills.csv`
- Handle the "cold start" problem with an onboarding survey or trending
  fallback for brand-new users
- Swap Cosine Similarity for Jaccard Similarity and compare results
- Turn this into a simple web app where users click skills instead of
  typing them

---
Built as part of the DecodeLabs AI Engineering Internship, 2026.
