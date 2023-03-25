from fastapi import FastAPI, File, UploadFile
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image ## pillow module
import tensorflow as tf
import requests
from starlette.middleware.cors import CORSMiddleware


app=FastAPI()

origins= [ "http://localhost:3000", "http://localhost", ]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods= ["*"],
	allow_headers= ["*"],
)
MODEL = tf.keras.models.load_model("../models/1")
class_names=['Early Blight','Late Blight','Healthy']
###method 1 to run ==> in terminal ' uvicorn main:app --reload'

@app.get("/ping")
async def ping():
    return "Hi this model is Made by Ashutosh tyagi"

def read_file_as_image(data) -> np.ndarray:
   # read data in bytes  ---> BytesIO(data)  
   ## Image.open(BytesIO(data)) open image from byte as pillow image
   image = np.array(Image.open(BytesIO(data)))
   return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,axis=0)  ## change 1d to 2d array
    
    prediction = MODEL.predict(img_batch)  ## cant take single image it take batch of image so image -> [ image ]


    ### np.argmax(prediction[0]) # return batch  return index of max value 0,1,2
    index = np.argmax(prediction[0]) # return batch  hence 0th index
    predict_class= class_names[index]
    confidence=round(np.max(prediction[0]),2)
    return {
        'class': predict_class,
        'confidence': float(confidence)
    }
if __name__=="__main__":
    uvicorn.run(app, host='localhost',port=8000)
