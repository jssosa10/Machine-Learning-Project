# Script para descargar los dataset externos en carpetas nuevas.
mkdir datasets
cd datasets
mkdir Caltech256
cd Caltech256
# obtener dataset de clasificación de imagenes
wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
cd ..
mkdir MSDGenere
cd MSDGenere
# obtener dataset de canciones.
wget http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/msd_genre_dataset.zip
# volver a la carpeta de origen. 
cd ..
cd ..