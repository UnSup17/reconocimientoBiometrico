from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from deepface import DeepFace  # Nueva importación
import cv2
import numpy as np
import tempfile
from typing import List

app = FastAPI()

# Cargar el clasificador Haar
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("No se pudo cargar el clasificador Haar de rostros.")

# Enum-like class
class FaceDetectionStatus:
    ONE_FACE = "ONE_FACE"
    ASPECT_RATIO_ONE_TO_ONE = "ASPECT_RATIO_ONE_TO_ONE"
    FACE_CLEAR = "FACE_CLEAR"

status_messages = {
    FaceDetectionStatus.ONE_FACE: {
        True: "Se detectó exactamente un rostro.",
        False: "La imagen debe contener exactamente un rostro."
    },
    FaceDetectionStatus.ASPECT_RATIO_ONE_TO_ONE: {
        True: "La imagen tiene una proporción de aspecto 1:1.",
        False: "La imagen debe tener una proporción de aspecto 1:1 (cuadrada)."
    },
    FaceDetectionStatus.FACE_CLEAR: {
        True: "El rostro es claro, de buen tamaño y está centrado.",
        False: "El rostro no está claro: asegúrese de que sea visible, centrado, no haga gestos y ocupe suficiente espacio en la imagen."
    }
}

# Pydantic models
class FaceDetectionStatusResult(BaseModel):
    status: str
    valid: bool
    message: str

class FaceDetectionResponse(BaseModel):
    statuses: List[FaceDetectionStatusResult]
    isValidPhoto: bool

@app.post("/faces/detect", response_model=FaceDetectionResponse)
async def detect_face(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")

        height, width = img.shape[:2]
        is_aspect_ratio_one_to_one = width == height

        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detección de rostros
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        has_one_face = len(faces) == 1
        is_face_clear = False
        is_face_centered = False
        is_real_face = False  # NUEVO

        if has_one_face:
            (x, y, w, h) = faces[0]

            face_area = w * h
            image_area = width * height
            face_big_enough = (face_area / image_area) > 0.15

            center_x = x + w // 2
            center_y = y + h // 2
            face_centered_h = width * 0.3 < center_x < width * 0.7
            face_centered_v = height * 0.3 < center_y < height * 0.7
            is_face_centered = face_centered_h and face_centered_v

            is_face_clear = face_big_enough and is_face_centered

            # --- Verificación de si es un rostro humano real ---
            try:
                DeepFace.verify(img1_path=img, img2_path=img, enforce_detection=True)
                is_real_face = True
            except:
                is_real_face = False

        is_valid_photo = has_one_face and is_aspect_ratio_one_to_one and is_face_clear and is_real_face

        statuses = [
            FaceDetectionStatusResult(
                status=FaceDetectionStatus.ONE_FACE,
                valid=has_one_face,
                message=status_messages[FaceDetectionStatus.ONE_FACE][has_one_face]
            ),
            FaceDetectionStatusResult(
                status=FaceDetectionStatus.ASPECT_RATIO_ONE_TO_ONE,
                valid=is_aspect_ratio_one_to_one,
                message=status_messages[FaceDetectionStatus.ASPECT_RATIO_ONE_TO_ONE][is_aspect_ratio_one_to_one]
            ),
            FaceDetectionStatusResult(
                status=FaceDetectionStatus.FACE_CLEAR,
                valid=is_face_clear,
                message=status_messages[FaceDetectionStatus.FACE_CLEAR][is_face_clear]
            ),
            FaceDetectionStatusResult(
                status="REAL_FACE",
                valid=is_real_face,
                message="Es un rostro humano real." if is_real_face else "No parece un rostro humano real."
            ),
        ]

        return FaceDetectionResponse(statuses=statuses, isValidPhoto=is_valid_photo)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen: {str(e)}")
