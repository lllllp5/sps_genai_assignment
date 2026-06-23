from fastapi import FastAPI, UploadFile, File
from app.cifar10_classifier import CIFAR10Classifier

app = FastAPI()

classifier = CIFAR10Classifier(model_path="cnn_model.pth")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    prediction = classifier.predict(image_bytes)
    return prediction

