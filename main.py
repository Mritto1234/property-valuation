from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vision_engine import AdvancedIngredientDetector
from recipe_engine import RecipeGenerator
import uvicorn

app = FastAPI(title="Enterprise Real-Time Fridge-to-Recipe API")

# Enable CORS for frontend web/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate engines once on application boot
# Delay heavy initialization until FastAPI startup
detector = None
recipe_bot = None

@app.on_event("startup")
def startup_event():
    global detector, recipe_bot
    detector = AdvancedIngredientDetector()
    recipe_bot = RecipeGenerator()

@app.post("/api/v1/scan-and-cook")
async def scan_and_cook(file: UploadFile = File(...)):
    # Verify file payload type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid image format. Provide JPEG or PNG.")

    try:
        # Read raw image payload
        image_bytes = await file.read()
        
        # 1. Run comprehensive detection (YOLO + Vision Fallback)
        all_detected_items = detector.scan_fridge(image_bytes)
        
        if not all_detected_items:
            return {
                "status": "success",
                "detected_count": 0,
                "ingredients": [],
                "recipes": [],
                "message": "No ingredients could be verified in the image."
            }
            
        # 2. Compile custom tailored recipes
        recipe_payload = recipe_bot.generate_menu(all_detected_items)
        
        return {
            "status": "success",
            "detected_count": len(all_detected_items),
            "ingredients": all_detected_items,
            "data": recipe_payload
        }

    except Exception as e:
        # Generic safety fallback for production runtime stability
        raise HTTPException(status_code=500, detail=f"Internal Execution Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)