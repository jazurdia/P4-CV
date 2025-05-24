import scipy.io
import matplotlib.pyplot as plt
import cv2

# Cargar el archivo .mat
mat = scipy.io.loadmat("./mall_dataset/mall_gt.mat")

# Extraer el diccionario de frames
frames = mat["frame"]

# Índice de la imagen (en MATLAB empieza en 1, en Python en 0)
img_index = 970

# Ruta de la imagen
img_path = f"./mall_dataset/frames/seq_{img_index:06d}.jpg"

# Leer imagen (usando OpenCV, que carga en BGR)
im = cv2.imread(img_path)
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Obtener coordenadas de las personas
# Nota: el array 'frame' está en formato [1, N], por eso usamos [0, img_index-1]
xy = frames[0, img_index - 1]["loc"][0, 0]

# Mostrar imagen y anotaciones
plt.imshow(im_rgb)
plt.scatter(xy[:, 0], xy[:, 1], color="red", marker="*")
plt.title(f"Frame {img_index}")
plt.axis("off")
plt.show()
