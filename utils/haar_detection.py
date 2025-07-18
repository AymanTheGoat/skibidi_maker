def getBoundingBoxsHaar(gray, face_cascade):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces_boxs = []
    
    img_height, img_width = gray.shape[:2]

    for (x, y, w, h) in faces:
        x1, y1 = int(x/1.2), int(y/5)
        x2, y2 = x1 + int(w*1.5), y1 + int(h*2)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)
        
        if x2 > x1 and y2 > y1:
            face = ((x1, y1), (x2, y2))
            faces_boxs.append(face)

    return faces_boxs