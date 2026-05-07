from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
  """
  Tai file model.pkl tu S3 ve may khi server khoi dong.
  Xac thuc thong qua IAM role tren EC2 hoac AWS credentials
  duoc cau hinh trong environment.
  """
  os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

  client = boto3.client("s3")
  client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)

  print(f"Model da duoc tai tu s3://{S3_BUCKET}/{S3_MODEL_KEY} ve {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
  features: list[float]


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
  """
  Dau vao : {"features": [f1, f2, ..., f12]}
  Dau ra  : {"prediction": 0|1|2, "label": "thap"|"trung_binh"|"cao"}
  """
  if len(req.features) != 12:
      raise HTTPException(
          status_code=400,
          detail="Expected 12 features (wine quality)",
      )

  pred = model.predict([req.features])[0]
  label_map = {
      0: "thap",
      1: "trung_binh",
      2: "cao",
  }

  return {
      "prediction": int(pred),
      "label": label_map[int(pred)],
  }


if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)