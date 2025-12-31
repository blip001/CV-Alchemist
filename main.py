import os
import uuid
import shutil
import logging
import tempfile
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel

import stripe
import vertexai
from google.auth import default
from vertexai.generative_models import GenerativeModel
from pdfminer.high_level import extract_text as extract_pdf
import docx
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

logging.basicConfig(level=logging.INFO)
PROJECT_ID = "cv-alchemist-482203"
LOCATION = "us-central1" 


app = FastAPI()
analysis_cache = {}

@app.on_event("startup")
def startup_event():
    # Load Key from Environment
    stripe.api_key = os.getenv("STRIPE_API_KEY")
    if not stripe.api_key:
        logging.warning("STRIPE_API_KEY environment variable not set.")
    try:
        credentials, discovered_project_id = default()
        # Use discovered project ID if available, otherwise fall back to the hardcoded one.
        project_to_use = discovered_project_id or PROJECT_ID
        if not project_to_use:
             raise Exception("Google Cloud project ID is not set or could not be discovered.")
        vertexai.init(project=project_to_use, location=LOCATION, credentials=credentials)
        logging.info("Vertex AI initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize Vertex AI: {e}")


app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def read_root():
    # Robustly find index.html
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    return HTMLResponse('<h1>Error: index.html not found</h1><p><a href="/">Go to Homepage</a></p>', status_code=500)

@app.get("/result/{result_id}")
async def get_result(result_id: str):
    """Retrieve a cached analysis result."""
    result = analysis_cache.get(result_id)
    if not result:
        return JSONResponse(status_code=404, content={"detail": "Result not found or expired."})
    return result

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), job_title: str = Form("")):
    text = ""
    temp_file_path = ""
    file_extension = os.path.splitext(file.filename)[1].lower()

    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Extract text based on file type
        if file_extension == ".pdf":
            text = extract_pdf(temp_file_path)
        elif file_extension == ".docx":
            doc = docx.Document(temp_file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return JSONResponse(status_code=400, content={"detail": "Unsupported file type. Please upload a PDF or DOCX."})

        # Call Vertex AI for analysis
        model = GenerativeModel("gemini-2.0-flash-lite-001")
        prompt = f"""Analyze the following resume for the position of '{job_title}'.
Provide a score and feedback based on its suitability for that specific role.
Resume Text: 
{text[:8000]}

Return your response as a JSON object with three keys: "score" (an integer between 0 and 100), "feedback" (a list of short, actionable feedback strings), and "raw_text_preview" (the first 2000 characters of the original text).
Example: {{"score": 85, "feedback": ["Great use of action verbs.", "Consider adding a summary section."], "raw_text_preview": "..."}}
"""
        response = model.generate_content(prompt)
        
        # Clean and parse the JSON response
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        analysis_result = json.loads(cleaned_response)

        # Cache the result
        result_id = str(uuid.uuid4())
        analysis_cache[result_id] = analysis_result
        analysis_result["result_id"] = result_id
        return analysis_result 

    except Exception as e:
        logging.error(f"Failed to analyze file: {e}")
        return JSONResponse(status_code=500, content={"detail": f"An error occurred during analysis: {e}"})
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/rewrite")
async def rewrite(req: dict):
    try:
        job_title = req.get('job_title', 'the user-specified position')
        model = GenerativeModel("gemini-2.0-flash-lite-001")
        prompt = f"Rewrite this resume to be highly optimized for the position of '{job_title}'. Return text only:\n{req.get('text','')[:8000]}"
        res = model.generate_content(prompt)
        return {"optimized_text": res.text.replace("```", "")}
    except Exception as e:
        logging.error(f"Error during rewrite: {e}")
        return {"optimized_text": f"Error: {str(e)}"}

@app.post("/create-checkout-session")
async def checkout(req: dict):
    if not stripe.api_key:
        logging.error("Stripe API key is not configured.")
        return JSONResponse(status_code=500, content={"detail": "Payment processor is not configured."})
    try:
        s = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{'price_data': {'currency': 'usd', 'product_data': {'name': 'Resume Optimization'}, 'unit_amount': 2999}, 'quantity': 1}],
            mode='payment',
            success_url=f"{req.get('origin_url')}?success=true",
            cancel_url=f"{req.get('origin_url')}?canceled=true",
            allow_promotion_codes=True,
            consent_collection={
                'promotions': 'auto',
            },
        )
        return {"url": s.url}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

class DL(BaseModel): text: str
@app.post("/download-pdf")
async def dl_pdf(r: DL): 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        f = temp_file.name
        c = canvas.Canvas(f, pagesize=letter)
        text_object = c.beginText(40, 750)
        text_object.setFont("Helvetica", 10)
        for line in r.text.split('\n'):
            text_object.textLine(line[:100])
        c.drawText(text_object)
        c.save()
    
    return FileResponse(f, filename="resume.pdf", background=lambda: os.remove(f))

@app.post("/download-docx")
async def dl_docx(r: DL):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
        f = temp_file.name
        d = docx.Document()
        d.add_paragraph(r.text)
        d.save(f)
    
    return FileResponse(f, filename="resume.docx", background=lambda: os.remove(f))

if __name__ == "__main__":
    import uvicorn
    # Use port 8080 to match common cloud shell configurations
    # The reload=True flag will automatically restart the server when you make code changes.
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)