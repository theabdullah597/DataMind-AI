from fastapi import FastAPI
app=FastAPI(title="Datamind API")
@app.get("/")
def root():
    return{
        "message":"Hello Datamind"
    }