from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from engine import calculate_career_scores

app = FastAPI()

# Enable CORS Policy so the frontend browser code can safely make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("career_model.pkl", "rb") as model_file:
    ml_model = pickle.load(model_file)

class StudentData(BaseModel):
    math: int
    physics: int
    chemistry: int
    cs: int
    coding_interest: int
    analytical_interest: int
    machine_interest: int
    building_interest: int
    electronics_interest: int
    logic_aptitude: int

class PremiumStudentData(StudentData):
    math_quiz_score: int
    logic_quiz_score: int
    domain_quiz_score: int

CAREER_ROADMAPS = {
    "Software Development": {
        "Year 1": "Master foundational programming (Python/C++) and Data Structures & Algorithms.",
        "Year 2": "Focus on Web Development architectures (Frontend & Backend frameworks) and databases.",
        "Year 3": "Build real-world full-stack projects, contribute to open-source, and apply for internships.",
        "Certifications": "AWS Certified Cloud Practitioner, Oracle Certified Java Professional."
    },
    "Data Science / AI": {
        "Year 1": "Learn Python, Advanced Linear Algebra, Probability, and Statistics.",
        "Year 2": "Master Data Manipulation (Pandas, NumPy) and core Machine Learning frameworks (Scikit-Learn).",
        "Year 3": "Deep Dive into Deep Learning (TensorFlow/PyTorch), Natural Language Processing, and portfolio building.",
        "Certifications": "Google Data Analytics Professional Certificate, IBM Data Science Professional."
    }
}

@app.post("/predict")
def get_career_recommendation(student: StudentData):
    student_dict = student.dict()
    
    # 1. Run Rule-Based Engine
    rule_results = calculate_career_scores(student_dict)
    top_rule_career = rule_results[0]["career"]
    
    # 2. Run Machine Learning Model
    input_features = [list(student_dict.values())]
    ml_prediction = ml_model.predict(input_features)[0]
    
    # Calculate ML Confidence
    probabilities = ml_model.predict_proba(input_features)[0]
    class_index = np.where(ml_model.classes_ == ml_prediction)[0][0]
    confidence_score = round(probabilities[class_index] * 100, 2)
    
    # 3. Hybrid Aggregator Decision Logic
    # If ML is confident, use it. If not, fallback to the transparent Rule Engine choice.
    if confidence_score >= 50.0:
        final_recommendation = ml_prediction
        decision_source = "Machine Learning Model (High Confidence)"
    else:
        final_recommendation = top_rule_career
        decision_source = "Rule Engine Override (Low ML Confidence Fallback)"
        
    return {
        "status": "success",
        "ml_recommended_career": final_recommendation,
        "ml_confidence": f"{confidence_score}% (Resolved via {decision_source})",
        "detailed_rankings": rule_results
    }

@app.post("/predict-premium")
def get_premium_recommendation(student: PremiumStudentData):
    student_dict = student.dict()
    quiz_features = ["math_quiz_score", "logic_quiz_score", "domain_quiz_score"]
    base_student_profile = {k: v for k, v in student_dict.items() if k not in quiz_features}
    
    base_student_profile["math"] = int((base_student_profile["math"] * 0.6) + (student_dict["math_quiz_score"] * 0.4))
    base_student_profile["logic_aptitude"] = int((base_student_profile["logic_aptitude"] * 0.6) + (student_dict["logic_quiz_score"] * 0.4))
    
    rule_results = calculate_career_scores(base_student_profile)
    input_features = [list(base_student_profile.values())]
    ml_prediction = ml_model.predict(input_features)[0]
    
    roadmap = CAREER_ROADMAPS.get(ml_prediction, {
        "Year 1": "Focus on core engineering engineering principles and math fundamentals.",
        "Year 2": "Specialize in your technical branch core projects and lab practicals.",
        "Year 3": "Work on an industry-grade capstone project and seek corporate internships.",
        "Certifications": "Relevant foundational industry vendor credentials."
    })
    
    return {
        "status": "premium_success",
        "verified_recommended_career": ml_prediction,
        "personalized_roadmap": roadmap
    }
def generate_ai_explanation(student_dict, recommended_career):
    """Generates natural language counseling insights based on student metrics."""
    strengths = []
    if student_dict["coding_interest"] >= 80: strengths.append("exceptional enthusiasm for coding")
    if student_dict["logic_aptitude"] >= 80: strengths.append("highly sharp logical reasoning capabilities")
    if student_dict["cs"] >= 80: strengths.append("a stellar academic foundation in Computer Science")
    if student_dict["physics"] >= 80: strengths.append("strong conceptual clarity in Physics")
    
    strength_text = ", combined with ".join(strengths) if strengths else "your unique technical profile"
    
    explanation = f"Based on our hybrid diagnostic review, **{recommended_career}** stands out as your optimal technical domain. " \
                  f"Our engines flagged your {strength_text} as key drivers for this choice. "
                  
    if student_dict["math"] < 50:
        explanation += "Note: Your baseline Mathematics marks are on the lower side. " \
                       "While your core logic and interest are fantastic for this field, we highly recommend focusing on practical discrete math modules during your first semester to stay ahead."
    else:
        explanation += "Your balanced academic scores indicate you are well-prepared to tackle the structural engineering coursework ahead."
        
    return explanation

@app.post("/predict")
def get_career_recommendation(student: StudentData):
    student_dict = student.dict()
    rule_results = calculate_career_scores(student_dict)
    top_rule_career = rule_results[0]["career"]
    
    input_features = [list(student_dict.values())]
    ml_prediction = ml_model.predict(input_features)[0]
    
    probabilities = ml_model.predict_proba(input_features)[0]
    class_index = np.where(ml_model.classes_ == ml_prediction)[0][0]
    confidence_score = round(probabilities[class_index] * 100, 2)
    
    if confidence_score >= 50.0:
        final_recommendation = ml_prediction
        decision_source = "Machine Learning Model (High Confidence)"
    else:
        final_recommendation = top_rule_career
        decision_source = "Rule Engine Override (Low ML Confidence Fallback)"
        
    # Generate the conversational AI text response
    ai_counseling_text = generate_ai_explanation(student_dict, final_recommendation)
        
    return {
        "status": "success",
        "ml_recommended_career": final_recommendation,
        "ml_confidence": f"{confidence_score}% ({decision_source})",
        "ai_explanation": ai_counseling_text,
        "detailed_rankings": rule_results
    }