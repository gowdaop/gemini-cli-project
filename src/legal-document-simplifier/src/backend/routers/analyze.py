from fastapi import APIRouter

router = APIRouter()

@router.post("/")
async def analyze_document():
    return {"message": "Document analyzed"}
