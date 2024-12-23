import cv2
import random
import math

# inisialisasi kamera
cap = cv2.VideoCapture(0)

#load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# dcctionary untuk menyimpan suhu tetap per wajah
face_temperatures = {}

def calculate_distance(face1, face2):
  
    x1, y1, w1, h1 = face1
    x2, y2, w2, h2 = face2
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # convert frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detect wajah dalam frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    new_face_temperatures = {}
    used_faces = set()
    
    for (x, y, w, h) in faces:
        current_face = (x, y, w, h)
        assigned_temperature = None
        
        # cari wajah lama yang paling dekat
        for old_face, temperature in face_temperatures.items():
            if old_face in used_faces:
                continue
            
            distance = calculate_distance(current_face, old_face)
            if distance < 50:  # Ambang batas jarak untuk menganggap wajah sama
                assigned_temperature = temperature
                used_faces.add(old_face)
                break
        
        # jika tidak ditemukan wajah lama yang cocok, buat suhu baru
        if assigned_temperature is None:
            assigned_temperature = random.randint(36, 39)  # rentang suhu normal hingga tinggi
        
        # simpan suhu wajah saat ini
        new_face_temperatures[current_face] = assigned_temperature
        
        # pixelation pada area wajah
        roi = frame[y:y+h, x:x+w]
        height, width = roi.shape[:2]
        
        # resize kecil lalu besar untuk efek pixelation
        roi = cv2.resize(roi, (width // 10, height // 10), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(roi, (width, height), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = roi
        
        # add kotak2 di muka
        color = (0, 255, 0)  # gree for normal
        if assigned_temperature > 38:
            color = (0, 0, 255)  # warnn (red)
            warning_text = "Bahaya! Cek ke dokter"
            cv2.putText(frame, warning_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
       
        cv2.putText(frame, f"{assigned_temperature}°C", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    
    face_temperatures = new_face_temperatures
    
    #  hasil
    cv2.imshow("Face Detection with Warning System", frame)
    
   # fc
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
