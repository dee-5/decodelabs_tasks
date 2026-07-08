"""
Project 3: AI Recommendation Logic — Tech Stack Recommender
DecodeLabs Industrial Training Kit - Batch 2026

Goal: Map a user's skills/interests to the most relevant tech career roles
using Content-Based Filtering (TF-IDF vectorization + Cosine Similarity).

Pipeline (Input -> Process -> Output):
  INPUT   -> Collect at least 3 user skills/interests
  PROCESS -> Convert skills into TF-IDF vectors, score similarity against
             each job role's skill profile
  OUTPUT  -> Sort by score, return the Top-3 recommended job roles
"""

import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------
# STEP 1 (INPUT): The "item" dataset — job roles and their required skills
# -------------------------------------------------
# In a production system this would live in a separate raw_skills.csv.
# We also generate that CSV below so it's included as a real dataset file.
JOB_ROLES = {
    "Data Scientist": "Python SQL Machine Learning Statistics Data Analysis Pandas",
    "DevOps Engineer": "AWS Docker Kubernetes CI/CD Linux Automation Cloud",
    "Backend Developer": "Java Python SQL APIs Databases Git Spring Boot",
    "Frontend Developer": "JavaScript HTML CSS React UI Design Web Design",
    "Cloud Architect": "AWS Azure Cloud Computing Networking Security Infrastructure",
    "Mobile Developer": "Swift Kotlin Android iOS Mobile UI App Development",
    "Data Engineer": "Python SQL ETL Spark Data Pipelines Big Data Airflow",
    "Cybersecurity Analyst": "Security Networking Penetration Testing Firewalls Linux",
    "UI/UX Designer": "Figma UI Design User Research Prototyping Wireframing",
    "Machine Learning Engineer": "Python Machine Learning TensorFlow Deep Learning Statistics Neural Networks",
}


def save_dataset_csv(filename="raw_skills.csv"):
    """Write the dataset out as a CSV file (the 'raw_skills.csv' from the brief)."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["job_role", "skills"])
        for role, skills in JOB_ROLES.items():
            writer.writerow([role, skills])


def get_user_skills():
    """
    STEP 1 (INPUT): Ingestion.
    Collect at least 3 skills/interests from the user (comma-separated).
    """
    print("=" * 55)
    print("TECH STACK RECOMMENDER")
    print("=" * 55)
    print("Enter at least 3 skills or interests (comma-separated).")
    print("Example: Python, Cloud Computing, Automation\n")

    raw = input("Your skills: ")
    skills = [s.strip() for s in raw.split(",") if s.strip()]

    while len(skills) < 3:
        print(f"Please enter at least 3 skills (you entered {len(skills)}).")
        raw = input("Your skills: ")
        skills = [s.strip() for s in raw.split(",") if s.strip()]

    return skills


def recommend(user_skills, job_roles, top_n=3):
    """
    STEP 2 (PROCESS): Vector Mapping + Scoring.
    Vectorize the user profile + all job roles with TF-IDF (so rare, specific
    skills are weighted higher than common ones), then rank job roles by
    Cosine Similarity to the user's profile.
    """
    roles = list(job_roles.keys())
    documents = list(job_roles.values())

    # The user's profile becomes just another "document" in the same
    # vocabulary space, so it can be compared fairly against every job role.
    user_profile = " ".join(user_skills)
    all_documents = documents + [user_profile]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_documents)

    # Last row = user vector, all rows before it = job role vectors
    user_vector = tfidf_matrix[-1]
    job_vectors = tfidf_matrix[:-1]

    # STEP 3 (PROCESS): Score every job role against the user vector
    scores = cosine_similarity(user_vector, job_vectors)[0]

    # STEP 4 (OUTPUT): Sort descending (Sorting), then keep only Top-N (Filtering)
    ranked = sorted(zip(roles, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def display_recommendations(ranked):
    print("\n" + "=" * 55)
    print("TOP RECOMMENDATIONS FOR YOU")
    print("=" * 55)

    if all(score == 0 for _, score in ranked):
        print("No strong matches found — try broader or different skills.")
        return

    for i, (role, score) in enumerate(ranked, start=1):
        match_percent = round(score * 100, 1)
        print(f"{i}. {role}  —  {match_percent}% match")


if __name__ == "__main__":
    save_dataset_csv()  # writes raw_skills.csv alongside the script
    user_skills = get_user_skills()
    top_matches = recommend(user_skills, JOB_ROLES, top_n=3)
    display_recommendations(top_matches)
