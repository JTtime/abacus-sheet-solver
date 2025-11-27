# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# from typing import List
# import cv2
# import numpy as np
# import pytesseract
# import re

# app = FastAPI(title="Math Worksheet Solver API")
# pytesseract.pytesseract.tesseract_cmd = r"C:\Users\tjeevraj\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"


# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173", "http://localhost:3000"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Pydantic Models
# class Problem(BaseModel):
#     problem_number: int = Field(..., description="Problem number (1-80)")
#     numbers: List[int] = Field(..., description="List of numbers to add/subtract")
#     answer: int = Field(..., description="Calculated sum")
#     row: int = Field(..., description="Row number (1-4)")
#     column: int = Field(..., description="Column number within row (1-20)")

# class WorksheetResponse(BaseModel):
#     total_problems: int
#     problems: List[Problem]
#     message: str
#     debug_info: dict = Field(default={}, description="Debug information")

# class WorksheetSolver:
    
#     def preprocess_image(self, image: np.ndarray) -> np.ndarray:
#         """Preprocess image for better OCR"""
#         # Convert to grayscale
#         if len(image.shape) == 3:
#             gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = image
        
#         # Apply adaptive thresholding
#         binary = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#             cv2.THRESH_BINARY, 11, 2
#         )
        
#         # Invert if needed (text should be black on white)
#         if np.mean(binary) < 127:
#             binary = cv2.bitwise_not(binary)
        
#         return binary
    
#     def extract_text_from_region(self, image: np.ndarray, x: int, y: int, w: int, h: int) -> str:
#         """Extract text from a specific region"""
#         # Add padding
#         padding = 5
#         y_start = max(0, y - padding)
#         y_end = min(image.shape[0], y + h + padding)
#         x_start = max(0, x - padding)
#         x_end = min(image.shape[1], x + w + padding)
        
#         roi = image[y_start:y_end, x_start:x_end]
        
#         if roi.size == 0:
#             return ""
        
#         # Preprocess ROI
#         if len(roi.shape) == 3:
#             roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
#         # Resize to improve OCR
#         scale = 3
#         roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
#         # Apply thresholding
#         _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         # OCR configuration
#         custom_config = r'--oem 3 --psm 6 -c tesseract_char_whitelist=0123456789-'
#         text = pytesseract.image_to_string(roi, config=custom_config)
        
#         return text.strip()
    
#     def parse_numbers_from_text(self, text: str) -> List[int]:
#         """Parse numbers from OCR text"""
#         numbers = []
        
#         # Split by newlines and process each line
#         lines = text.split('\n')
        
#         for line in lines:
#             # Clean the line
#             line = line.strip()
#             line = re.sub(r'[^\d\-]', '', line)  # Keep only digits and minus
            
#             if not line:
#                 continue
            
#             # Handle multiple numbers on same line
#             # Split by spaces or find all number patterns
#             number_patterns = re.findall(r'-?\d+', line)
            
#             for num_str in number_patterns:
#                 try:
#                     num = int(num_str)
#                     # Reasonable range check
#                     if -1000 < num < 1000:
#                         numbers.append(num)
#                 except ValueError:
#                     continue
        
#         return numbers
    
#     def solve_worksheet(self, image: np.ndarray) -> tuple:
#         """Solve worksheet using grid-based approach"""
#         height, width = image.shape[:2]
        
#         # Define margins and layout
#         # Based on the image structure
#         top_margin = int(height * 0.12)  # Skip header
#         bottom_margin = int(height * 0.02)
#         left_margin = int(width * 0.08)  # Skip left category label
#         right_margin = int(width * 0.02)
        
#         # Calculate usable area
#         usable_height = height - top_margin - bottom_margin
#         usable_width = width - left_margin - right_margin
        
#         # 4 rows, 20 columns
#         row_height = usable_height // 4
#         col_width = usable_width // 20
        
#         problems = []
#         problem_number = 1
        
#         debug_info = {
#             "image_size": f"{width}x{height}",
#             "usable_area": f"{usable_width}x{usable_height}",
#             "cell_size": f"{col_width}x{row_height}",
#             "detected_problems": 0
#         }
        
#         for row_idx in range(4):
#             for col_idx in range(20):
#                 # Calculate cell position
#                 x = left_margin + (col_idx * col_width)
#                 y = top_margin + (row_idx * row_height)
                
#                 # Extract text from this cell
#                 text = self.extract_text_from_region(image, x, y, col_width, row_height)
                
#                 # Parse numbers
#                 numbers = self.parse_numbers_from_text(text)
                
#                 # Create problem if we have numbers
#                 if len(numbers) >= 2:  # At least 2 numbers
#                     answer = sum(numbers)
                    
#                     problem = Problem(
#                         problem_number=problem_number,
#                         numbers=numbers,
#                         answer=answer,
#                         row=row_idx + 1,
#                         column=col_idx + 1
#                     )
#                     problems.append(problem)
#                     debug_info["detected_problems"] += 1
                
#                 problem_number += 1
        
#         return problems, debug_info
    
#     def detect_and_solve_by_contours(self, image: np.ndarray) -> tuple:
#         """Alternative method: detect boxes by contours"""
#         # Preprocess
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                        cv2.THRESH_BINARY_INV, 11, 2)
        
#         # Find contours
#         contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
#         # Filter for rectangular boxes
#         boxes = []
#         for contour in contours:
#             x, y, w, h = cv2.boundingRect(contour)
#             aspect_ratio = w / float(h) if h > 0 else 0
            
#             # Filter by size and aspect ratio
#             if 30 < w < 150 and 200 < h < 400 and 0.2 < aspect_ratio < 0.8:
#                 boxes.append((x, y, w, h))
        
#         # Sort boxes by position (left to right, top to bottom)
#         boxes = sorted(boxes, key=lambda b: (b[1] // 200, b[0]))
        
#         problems = []
#         problem_number = 1
        
#         for box_idx, (x, y, w, h) in enumerate(boxes):
#             text = self.extract_text_from_region(image, x, y, w, h)
#             numbers = self.parse_numbers_from_text(text)
            
#             if len(numbers) >= 2:
#                 answer = sum(numbers)
                
#                 # Estimate row and column
#                 row = (y // (image.shape[0] // 5)) + 1
#                 col = (box_idx % 20) + 1
                
#                 problem = Problem(
#                     problem_number=problem_number,
#                     numbers=numbers,
#                     answer=answer,
#                     row=min(row, 4),
#                     column=col
#                 )
#                 problems.append(problem)
            
#             problem_number += 1
        
#         debug_info = {
#             "method": "contour_detection",
#             "boxes_found": len(boxes),
#             "problems_solved": len(problems)
#         }
        
#         return problems, debug_info

# solver = WorksheetSolver()

# @app.post("/solve-worksheet", response_model=WorksheetResponse)
# async def solve_worksheet(file: UploadFile = File(...)):
#     """
#     Upload a math worksheet image and get the solutions.
#     Expected format: 80 problems (4 rows × 20 columns)
#     """
#     try:
#         # Validate file type
#         if not file.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read image
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image file")
        
#         # Try grid-based method first
#         problems, debug_info = solver.solve_worksheet(image)
        
#         # If we don't get enough problems, try contour method
#         if len(problems) < 40:
#             problems_alt, debug_info_alt = solver.detect_and_solve_by_contours(image)
#             if len(problems_alt) > len(problems):
#                 problems = problems_alt
#                 debug_info.update(debug_info_alt)
        
#         if not problems:
#             raise HTTPException(
#                 status_code=422,
#                 detail=f"Could not detect problems. Debug: {debug_info}"
#             )
        
#         return WorksheetResponse(
#             total_problems=len(problems),
#             problems=problems,
#             message=f"Successfully solved {len(problems)} problems",
#             debug_info=debug_info
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Error: {str(e)}\n{traceback.format_exc()}"
#         )

# @app.post("/test-ocr")
# async def test_ocr(file: UploadFile = File(...)):
#     """
#     Test endpoint to see what OCR is extracting from the entire image
#     """
#     try:
#         contents = await file.read()
#         nparr = np.frombuffer(contents, np.uint8)
#         image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Preprocess
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#         # Get full text
#         full_text = pytesseract.image_to_string(gray)
        
#         # Get text with bounding boxes
#         data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        
#         detected_numbers = []
#         for i, text in enumerate(data['text']):
#             if text.strip() and data['conf'][i] > 0:
#                 detected_numbers.append({
#                     "text": text,
#                     "confidence": data['conf'][i],
#                     "position": {
#                         "x": data['left'][i],
#                         "y": data['top'][i],
#                         "w": data['width'][i],
#                         "h": data['height'][i]
#                     }
#                 })
        
#         return {
#             "full_text_preview": full_text[:500],
#             "detected_items": len(detected_numbers),
#             "sample_detections": detected_numbers[:20]
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {
#         "message": "Math Worksheet Solver API",
#         "endpoints": {
#             "/solve-worksheet": "POST - Get JSON response with all answers (80 problems expected)",
#             "/test-ocr": "POST - Test OCR extraction to debug issues"
#         }
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)