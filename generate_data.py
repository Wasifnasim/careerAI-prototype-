# 
import pandas as pd
import numpy as np
import random

# Puraane code se careers ki list le li
CAREERS = [
    "Software Development", "Data Science / AI", "Cybersecurity", "Cloud / DevOps",
    "Mechanical Engineering", "Civil Engineering", "Electrical Engineering",
    "Robotics / Automation", "Electronics & Communication"
]

def generate_student_dataset(num_records=5000):
    np.random.seed(42) # Taaki har baar same random data bane
    random.seed(42)
    
    data = []
    
    for _ in range(num_records):
        # 1. Random Marks (Scale: 0 to 100) - PCM aur CS ke marks
        math = int(np.clip(np.random.normal(75, 15), 40, 100))
        physics = int(np.clip(np.random.normal(72, 15), 40, 100))
        chemistry = int(np.clip(np.random.normal(70, 15), 40, 100))
        cs = int(np.clip(np.random.normal(75, 18), 40, 100))
        
        # 2. Random Interests & Aptitude (Scale: 0 to 100)
        coding_interest = random.randint(10, 100)
        analytical_interest = random.randint(10, 100)
        machine_interest = random.randint(10, 100)
        building_interest = random.randint(10, 100)
        electronics_interest = random.randint(10, 100)
        logic_aptitude = random.randint(30, 100)
        
        # 3. LOGIC: Bache ke data ke hisab se sahi career label chunna
        # Agar coding aur math dono acche hain toh Tech careers milne chahiye
        if coding_interest > 75 and math > 70:
            target_career = random.choice(["Software Development", "Data Science / AI", "Cybersecurity", "Cloud / DevOps"])
        elif machine_interest > 70 and physics > 70:
            target_career = random.choice(["Mechanical Engineering", "Robotics / Automation"])
        elif electronics_interest > 70 and math > 65:
            target_career = random.choice(["Electrical Engineering", "Electronics & Communication"])
        elif building_interest > 70 and physics > 65:
            target_career = "Civil Engineering"
        else:
            # Agar bacha mix hai ya confuse hai, toh koi bhi random career
            target_career = random.choice(CAREERS)
            
        # Pura data ek row mein save kar rahe hain
        student_record = {
            "math": math,
            "physics": physics,
            "chemistry": chemistry,
            "cs": cs,
            "coding_interest": coding_interest,
            "analytical_interest": analytical_interest,
            "machine_interest": machine_interest,
            "building_interest": building_interest,
            "electronics_interest": electronics_interest,
            "logic_aptitude": logic_aptitude,
            "target_career": target_career  # Yeh humara Output/Label hai
        }
        data.append(student_record)
        
    # Data ko Table (DataFrame) mein convert karke CSV file bana do
    df = pd.DataFrame(data)
    df.to_csv("student_careers_dataset.csv", index=False)
    print(f"✅ Success: 5,000 students ka data 'student_careers_dataset.csv' mein save ho gaya hai!")

if __name__ == "__main__":
    generate_student_dataset()