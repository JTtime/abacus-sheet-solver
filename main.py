from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
from io import BytesIO

app = FastAPI(title="Abacus Competition Checker")

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\tjeevraj\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionResult(BaseModel):
    question_number: int
    row: int
    position_in_row: int
    numbers: List[int]
    correct_answer: int
    attempted_answer: Optional[int]
    is_correct: bool

class CheckResult(BaseModel):
    total_questions: int
    correct_answers: int
    wrong_answers: int
    unattempted: int
    accuracy_percentage: float
    results: List[QuestionResult]

# Hardcoded problems data based on the image structure
PROBLEMS_DATA = [
    # Row 1 (Questions 1-20) - 3 numbers each
    [[7, 8, -5], [9, -9, 3], [9, 8, -5], [8, -1, 4], [9, -4, -5], [18, 4, 1], [8, 4, -3], [8, -8, 3], [8, -5, 9], [9, 8, 8], [8, 9, -8], [10, -9, 5], [8, 7, 3], [9, -1, -2], [1, 29, -8], [1, 6, -7], [8, 1, -7], [2, 2, 9], [6, 3, -1], [18, 7, -6]],
    
    # Row 2 (Questions 21-40) - 3 numbers each
    [[4, 5, 6], [1, 5, 4], [2, -1, 5], [7, 2, 5], [7, 8, 5], [8, 2, -5], [1, 8, 1], [4, 6, -5], [2, 8, 1], [5, 1, 1], [6, 3, -1], [2, 2, 6], [1, 1, 1], [3, 8, 1], [1, 2, 7], [4, 7, -1], [5, 2, 1], [9, 1, 1], [2, 1, 0], [19, 1, -5]],
    
    # Row 3 (Questions 41-60) - 4 numbers each
    [[7, 8, 4, 9], [3, 7, 8, -7], [23, 6, 3, -2], [7, 2, 7, -6], [73, 1, 5, -9], [4, 8, -1, 9], [3, 7, 7, 2], [34, 7, 5, 2], [7, 2, 6, -5], [9, 7, -6, 9], [2, 8, 4, 5], [29, -3, 7, 6], [2, 9, 8, -5], [36, 3, 1, 9], [3, 9, 7, -8], [39, 2, 1, 5], [2, 7, 7, 4], [9, 6, 3, 3], [83, 6, -9, 3], [8, 9, -3, 9]],
    
    # Row 4 (Questions 61-80) - 4 numbers each
    [[7, -5, 9, 8], [3, 1, 7, 3], [4, -1, 8, 6], [2, 2, 9, 7], [8, -5, 8, 7], [2, -1, 9, 7], [9, 9, 4, 2], [8, 4, 8, 4], [4, 5, 3, 1], [7, -5, 9, 8], [10, 5, 4, 3], [6, 1, -8, 3], [9, 7, 5, 5], [17, -5, 7, 2], [9, -5, 8, 5], [9, -5, 7, 2], [7, 4, 9, 8], [9, -3, 5, 9], [9, 5, 5, 5], [15, 3, 8, 4]],
    
    # Row 5 (Questions 81-100) - 5 numbers each
    [[3, 11, -5, 9, 1], [7, 9, 22, -11, 8], [9, 0, 11, 6, 4], [9, 11, 4, 5, 8], [8, 3, 9, 78, -13], [8, 2, 8, 2, 8], [3, 7, 22, -2, 3], [9, -5, 8, 5, 5], [6, 4, 8, 7, 3], [4, -2, 7, 5, 5], [19, 4, -3, 6, 3], [9, 2, 8, 2, 6], [2, 7, 33, -2, 9], [56, 5, 7, 5, 5], [12, 9, 5, 5, 7], [17, 2, 5, 6, 5], [6, 4, 9, 4, 7], [9, -4, 24, 4, 6], [9, -2, 4, 3, 6], [6, 3, 7, 12, 5]]
]

def calculate_answer(numbers: List[int]) -> int:
    """Calculate the sum of numbers"""
    return sum(numbers)

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocess image for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh)
    
    return denoised

def extract_attempted_answers(image_bytes: bytes) -> List[Optional[int]]:
    """Extract attempted answers from the image using OCR"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        # Preprocess image
        processed = preprocess_image(img)
        
        # Get image dimensions
        height, width = processed.shape
        
        # Define approximate regions for each row's answer boxes
        # These are rough estimates and may need adjustment based on actual image
        row_regions = [
            (int(height * 0.12), int(height * 0.23)),  # Row 1
            (int(height * 0.25), int(height * 0.36)),  # Row 2
            (int(height * 0.38), int(height * 0.49)),  # Row 3
            (int(height * 0.51), int(height * 0.62)),  # Row 4
            (int(height * 0.64), int(height * 0.75)),  # Row 5
        ]
        
        attempted_answers = []
        question_num = 0
        
        for row_idx, (y_start, y_end) in enumerate(row_regions):
            # Extract row region
            row_img = processed[y_start:y_end, :]
            
            # Divide row into 20 equal columns
            col_width = width // 20
            
            for col_idx in range(20):
                x_start = col_idx * col_width
                x_end = (col_idx + 1) * col_width
                
                # Extract cell (answer box area - bottom portion)
                cell_height = row_img.shape[0]
                answer_region = row_img[int(cell_height * 0.7):, x_start:x_end]
                
                # OCR on the answer region
                custom_config = r'--oem 3 --psm 7 -c tesseract_char_whitelist=0123456789-'
                text = pytesseract.image_to_string(answer_region, config=custom_config)
                
                # Clean and parse the text
                text = text.strip().replace(' ', '').replace('O', '0')
                
                # Try to extract number
                if text:
                    try:
                        # Handle negative numbers
                        match = re.search(r'-?\d+', text)
                        if match:
                            attempted_answers.append(int(match.group()))
                        else:
                            attempted_answers.append(None)
                    except:
                        attempted_answers.append(None)
                else:
                    attempted_answers.append(None)
                
                question_num += 1
        
        return attempted_answers
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def check_answers(attempted_answers: List[Optional[int]]) -> CheckResult:
    """Check attempted answers against correct answers"""
    results = []
    correct_count = 0
    wrong_count = 0
    unattempted_count = 0
    
    question_num = 1
    
    for row_idx, row_problems in enumerate(PROBLEMS_DATA):
        for pos_idx, numbers in enumerate(row_problems):
            correct_answer = calculate_answer(numbers)
            attempted_answer = attempted_answers[question_num - 1] if question_num - 1 < len(attempted_answers) else None
            
            is_correct = False
            if attempted_answer is None:
                unattempted_count += 1
            elif attempted_answer == correct_answer:
                is_correct = True
                correct_count += 1
            else:
                wrong_count += 1
            
            result = QuestionResult(
                question_number=question_num,
                row=row_idx + 1,
                position_in_row=pos_idx + 1,
                numbers=numbers,
                correct_answer=correct_answer,
                attempted_answer=attempted_answer,
                is_correct=is_correct
            )
            results.append(result)
            question_num += 1
    
    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    return CheckResult(
        total_questions=total,
        correct_answers=correct_count,
        wrong_answers=wrong_count,
        unattempted=unattempted_count,
        accuracy_percentage=round(accuracy, 2),
        results=results
    )

@app.post("/check-answers", response_model=CheckResult)
async def check_abacus_answers(file: UploadFile = File(...)):
    """
    Upload an abacus competition answer sheet image and get results.
    
    The image should contain:
    - Row 1-2: 20 questions each with 3 numbers (Questions 1-40)
    - Row 3-4: 20 questions each with 4 numbers (Questions 41-80)
    - Row 5: 20 questions with 5 numbers (Questions 81-100)
    """
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Extract attempted answers from image
        attempted_answers = extract_attempted_answers(image_bytes)
        
        # Check answers
        result = check_answers(attempted_answers)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Abacus Competition Checker API",
        "version": "1.0",
        "endpoints": {
            "/check-answers": "POST - Upload answer sheet image",
            "/docs": "API documentation"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

# For testing purposes - manually input attempted answers
@app.post("/check-answers-manual", response_model=CheckResult)
async def check_answers_manual(attempted_answers: List[Optional[int]]):
    """
    Manually provide attempted answers as a list of 100 integers (or null for unattempted).
    Useful for testing without OCR.
    """
    if len(attempted_answers) != 100:
        raise HTTPException(status_code=400, detail="Must provide exactly 100 answers")
    
    result = check_answers(attempted_answers)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)