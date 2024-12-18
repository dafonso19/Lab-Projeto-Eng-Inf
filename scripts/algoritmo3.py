import cv2
import numpy as np
import os


total_defects = 0
defects = []
defect_confidences = []
defect_labels = []

def detect_defects(raw_image, lane_mask):
    # Convertendo a imagem para tons de cinza
    gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    
    # Aplicando thresholding para binarizar a imagem
    #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask_tresh = cv2.adaptiveThreshold(gray,254,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,621,3)
    
    #thresh_inv = cv2.adaptiveThreshold(gray,254,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,65,-5)
    #masked_image_inv = cv2.bitwise_or(thresh_inv, thresh_inv, mask=lane_mask)
    #cv2.imshow('INVERSSE Mask', masked_image_inv)
    # Aplicando a máscara da LANE
    #Define os bits brancos da lane e compara com a mask_tresh
    masked_image = cv2.bitwise_and(mask_tresh, mask_tresh, mask=lane_mask)
    
    
    cv2.imshow('Markings Mask', masked_image)
    # Encontrando os contornos na imagem binarizada
    contours, _ = cv2.findContours(masked_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    

    # Inicializando lista de defeitos
    defects = []

    # Identificando os contornos que correspondem aos defeitos
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1500 and area < 70000:
            # Verificar o tipo de defeito com base em critérios como forma, tamanho, etc.
            # Adicionar o contorno à lista de defeitos
            confidence = calculate_defect_confidence(area, threshold_defeito_grande)
            if area < threshold_defeito_grande:
                defects.append((cnt, "Defeito Grande", confidence))
            elif area < desgaste_threshold:
                defects.append((cnt, "Desgaste", confidence))
            elif area < deformacao_threshold:
                defects.append((cnt, "Deformação", confidence))
            elif area < buracos_threshold:
                # Verificar se o contorno está dentro de outro contorno. Se sim, é um buraco
                #if is_contour_inside_another(cnt, contours):
                    defects.append((cnt, "Buraco", confidence))
                #else:
                #    defects.append((cnt, "Defeito Pequeno", confidence))
            else:
                defects.append((cnt, "Remendo", confidence))
    
    # Adicionar os defeitos e as respectivas confianças às listas globais
    for defect in defects:
        cnt, label, confidence = defect
        defect_labels.append(label)
        defect_confidences.append(confidence)

    # Desenhar os contornos na imagem original e adicionar legendas
    for defect in defects:
        cnt, label, _ = defect
        color = get_defect_color(label)
        cv2.drawContours(raw_image, [cnt], -1, color, 2)
        cv2.putText(raw_image, label, (cnt[0][0][0], cnt[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return raw_image

def is_contour_inside_another(contour, contours):
    for cnt in contours:
        if cv2.contourArea(cv2.convexHull(contour)) < cv2.contourArea(cv2.convexHull(cnt)) and cv2.pointPolygonTest(cnt, tuple(contour[0][0]), False) >= 0:
            return True
    return False

def get_defect_color(label):
    if label == "Defeito Pequeno":
        return (0, 0, 255)  # vermelho
    elif label == "Defeito Grande":
        return (255, 0, 0)  # azul
    elif label == "Desgaste":
        return (255, 255, 0)  # amarelo
    elif label == "Deformação":
        return (127, 127, 64)  # cinza
    elif label == "Buraco":
        return (0, 0, 0)  # preto
    elif label == "Remendo":
        return (255, 0, 255)  # magenta


def calculate_defect_confidence(area, threshold):
    if area > threshold:
        return (area - threshold) / (70000 - threshold)
    else:
        return 0

#threshold_defeito_grande = 70000
#deformacao_threshold = 12000
#desgaste_threshold = 5000
#buracos_threshold = 2000

threshold_defeito_grande = 1500
deformacao_threshold = 5000
desgaste_threshold = 12000
buracos_threshold = 70000

data_dir = '/Users/afonso/Documents/LAB_PROJETO/Projeto final/Dataset1/'

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)

    # Verifique se é uma pasta válida
    if os.path.isdir(folder_path):
        raw_images = []  # Lista para armazenar as imagens RAW
        lane_masks = []  # Lista para armazenar as máscaras da LANE

        # Percorra os arquivos dentro da pasta
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Verifique o final do nome do arquivo para identificar as imagens RAW
            if file_name.endswith('RAW.jpg'):
                # Carregue a imagem RAW
                raw_image = cv2.imread(file_path)
                raw_images.append(raw_image)

            # Verifique o final do nome do arquivo para identificar as máscaras da LANE
            if file_name.endswith('LANE.png'):
                # Carregue a máscara da LANE
                lane_mask = cv2.imread(file_path, 0)
                lane_masks.append(lane_mask)

        # Execute o código de detecção de defeitos nas imagens RAW
        for raw_image, lane_mask in zip(raw_images, lane_masks):
            img_with_defects = detect_defects(raw_image, lane_mask)
            cv2.imshow('image', img_with_defects)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

defect_types = [ "Desgaste", "Buraco"]
total_images = len(raw_images)
total_defects = len(defect_labels)

print("Defect Detection Results:")
print("=======================================")
print("Defect Type\t\tDefect Count\tConfidence (%)")
print("---------------------------------------")

for defect_type in defect_types:
    defect_count = defect_labels.count(defect_type)
    confidence_sum = sum([defect_confidences[i] for i, label in enumerate(defect_labels) if label == defect_type])
    confidence_percentage = (confidence_sum / total_defects) * 100 if total_defects > 0 else 0

    print(f"{defect_type}\t\t{defect_count}\t\t{confidence_percentage:.2f}")

print("---------------------------------------")
print(f"Total Defects:\t\t{total_defects}\t\t100.00")

cv2.destroyAllWindows()
