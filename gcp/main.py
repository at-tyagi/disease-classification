from google.cloud import storage 
import tensorflow as tf 
from PIL import Image
import numpy as np 

BUCKET_NAME = "disease-classification-model2"
class_names = ["Early Blight", " Late Blight", "Healthy "]
model = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

def predict(request):
    global model
    if model is None:
        print("will download model")
        download_blob(
            BUCKET_NAME,
            "models/model2.h5",
            "/tmp/model2.h5",
        )
        model = tf.keras.models.load_model("/tmp/model2.h5")
        print("model downloaded")

    image = request.files["files"]
    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image = image/255
    img_array = tf.expand_dims(image,0)
    prediction = model.predict(img_array)
    print("prediction  : ", prediction)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(100 * (np.max(prediction[0])), 2)
    return {"class": predicted_class, "confidence": confidence}
