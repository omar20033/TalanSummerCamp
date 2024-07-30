from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from pprint import pprint
from vector import app




# Initialize FastAPI app    
api_app = FastAPI()

# Define request model
class QueryRequest(BaseModel):
    question: str

# Define response model
class QueryResponse(BaseModel):
    question: str
    generation: str

# Define your workflow application (as shown in your workflow)

@api_app.post("/query", response_model=QueryResponse)
async def query_rag_model(request: QueryRequest):
    try:
        inputs = {"question": request.question}
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint(f"Finished running: {key}:")
        response = value["generation"]  
        return QueryResponse(question=request.question, generation=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main function to run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api_app, host="localhost", port=8000)