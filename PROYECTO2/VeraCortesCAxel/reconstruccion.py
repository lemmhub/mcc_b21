# -*- coding: utf-8 -*-
"""
@author: Axel Vera
"""

import cv2
import os
import sys
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
import open3d as o3d

#Función para la calibración de la cámara
def calibrate(chessboard_size):
    # chessboard_size = (9,6)
    if os.path.exists('calibration_images'):
        obj_points = [] 
        img_points = [] 
        
        #Preparando los puntos
        objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
        
        #Leer imagenes    
        calibration_paths = glob.glob('calibration_images/*')
        
        #Iteración sobre las imágenes para encontrar la matriz intrínseca
        for image_path in tqdm(calibration_paths):
        	#Cargar imagen
        	image = cv2.imread(image_path)
        	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        	print("Image loaded, Analizying...")
        	#Buscar esquinas del tablero
        	ret,corners = cv2.findChessboardCorners(gray_image, chessboard_size, None)
        
        	if ret == True:
        		print("Chessboard detected!")
        		print(image_path)
        		#Criterio de precisión del píxel
        		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        		#Refinar la ubiación de la esquina acorde al criterio
        		cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
        		obj_points.append(objp)
        		img_points.append(corners)
        
        #Calibración
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,gray_image.shape[::-1], None, None)
    else:
        os.mkdir('calibration_images')
        print('Es necesario introducir imágnes de muestra en la carpeta calibration_images')
        sys.exit()
        
    #Guardar parámetros
    np.save("camera_params/ret", ret)
    np.save("camera_params/K", K)
    np.save("camera_params/dist", dist)
    np.save("camera_params/rvecs", rvecs)
    np.save("camera_params/tvecs", tvecs)
    
    #Obtener exif data
    exif_img = PIL.Image.open(calibration_paths[0])
    
    exif_data = {
    	PIL.ExifTags.TAGS[k]:v
    	for k, v in exif_img._getexif().items()
    	if k in PIL.ExifTags.TAGS}
    
    #Obtener y guardar focal length
    focal_length = exif_data['FocalLength']
    np.save("camera_params/FocalLength", focal_length)
    
#Función para crear nube de puntos
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

#Función para submuestrear imágenes
def downsample_image(image, reduce_factor):
	for i in range(0,reduce_factor):
		if len(image.shape) > 2:
			row,col = image.shape[:2]
		else:
			row,col = image.shape

		image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
	return image


#Load camera parameters

if os.path.exists("camera_params"):
    try:
        ret = np.load('camera_params/ret.npy')
        K = np.load('camera_params/K.npy')
        dist = np.load('camera_params/dist.npy')
    except:
        print('No existen parámetros. Es necesario hacer una calibración')
        print('Introduce ancho del patrón de tablero de ajedrez:')
        w = int(input())
        print('Introduce ancho del patrón de tablero de ajedrez:')
        h = int(input())
        calibrate((w,h))
        ret = np.load('camera_params/ret.npy')
        K = np.load('camera_params/K.npy')
        dist = np.load('camera_params/dist.npy')
else:
    os.mkdir("camera_params")
    print('No existen parámetros. Es necesario hacer una calibración')
    print('Introduce ancho del patrón de tablero de ajedrez:')
    w = input()
    print('Introduce ancho del patrón de tablero de ajedrez:')
    h = input()
    calibrate((w,h))
    ret = np.load('camera_params/ret.npy')
    K = np.load('camera_params/K.npy')
    dist = np.load('camera_params/dist.npy')


print('Introduce el nombre de la imagen izquierda:')
img_path1 = input()
print('Introduce el nombre de la imagen derecha:')
img_path2 = input()

if not os.path.exists(img_path1) or not os.path.exists(img_path2):
    print('Al parecer no existen las imágenes. Colocalas en la misma carpeta que este archivo')
    sys.exit()

#Cargar imágenes
img_1 = cv2.imread(img_path1)
img_2 = cv2.imread(img_path2)

#Obtener alto y ancho
h,w = img_2.shape[:2]

#Obtener matriz óptrima para reducir la distorción
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

#Reducción de la distorción
img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

#Submuestreo de imágenes
img_1_downsampled = downsample_image(img_1_undistorted,3)
img_2_downsampled = downsample_image(img_2_undistorted,3)

#Parámetros de disparidad
win_size = 5
min_disp = 1
max_disp = 17
num_disp = max_disp - min_disp # Tiene que ser divisible entre 16

#Objeto BM
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
	numDisparities = num_disp,
	blockSize = 5,
	uniquenessRatio = 5,
	speckleWindowSize = 5,
	speckleRange = 5,
	disp12MaxDiff = 2,
	P1 = 8*3*win_size**2,
	P2 =32*3*win_size**2)

#Calcular mapa de disparidad
print ("\nCalculando el mapa de disparidad...")
disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

#Crear nube de puntos
print ("\nGenerando map 3D...")

#Ancho y alto de la imagen submuestreada
h,w = img_2_downsampled.shape[:2]

#Cargar focal length
focal_length = np.load('camera_params/FocalLength.npy', allow_pickle=True)

#Matriz de transformación. Obtenidad de:
#Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,focal_length*0.05,0],
				[0,0,0,1]])

#Proyectar puntos en 3D
points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
#Obtener color de los puntos
colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)

#Deshacerse de puntos con valor 0 (sin profundidad)
mask_map = disparity_map > disparity_map.min()

#Máscara de colores y puntos
output_points = points_3D[mask_map]
output_colors = colors[mask_map]

#Nombre del archivo de salida
output_file = 'reconstructed.ply'

#Generate point cloud 
print ("\n Creando el archivo de salida... \n")
create_output(output_points, output_colors, output_file)

#Mostrar la reconstrucción en 3D
pcd = o3d.io.read_point_cloud("reconstructed.ply")
o3d.visualization.draw_geometries([pcd])
