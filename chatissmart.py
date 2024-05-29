import cv2
import numpy as np
from matplotlib import pyplot as plt
def remove_black_pixels_inside_ring(image_path, output_path):
    # Bild einlesen
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    # In Graustufen konvertieren
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.subplot(133),plt.hist(gray_image.ravel(),256,[0,256])
    # Schwellenwert für die binäre Maske setzen
    _, mask = cv2.threshold(gray_image, 1, 50, cv2.THRESH_BINARY)
    plt.subplot(132),plt.plot(mask)
    
    # Konturen finden
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Größte Kontur (den Ring) finden
    max_contour = max(contours, key=cv2.contourArea)
    
    # Eine Maske des Rings erstellen
    ring_mask = np.zeros_like(mask)
    cv2.drawContours(ring_mask, [max_contour], -1, (255), thickness=cv2.FILLED)
    
    # Nur den Bereich innerhalb des Rings beibehalten
    ring_mask = cv2.bitwise_and(mask, ring_mask)
    
    # Farbmaske anwenden, um die schwarzen Pixel innerhalb des Rings zu entfernen
    result_image = cv2.bitwise_and(image, image, mask=ring_mask)
    
    # Schwarze Pixel innerhalb des Rings durch Weiß ersetzen
    result_image[np.where((result_image == [0, 0, 0]).all(axis=2))] = [255, 255, 255]
    
    # Ergebnisbild speichern
    cv2.imwrite(output_path, result_image)

# Beispielverwendung
input_image_path = 'pic.jpg'

output_image_path = 'output_image.png'
remove_black_pixels_inside_ring(input_image_path, output_image_path)
