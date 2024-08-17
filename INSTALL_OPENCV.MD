# Compilando o OpenCV para habilitar o opencv_contrib

Para utilizar os algoritmos SURF e BRIEF, que são detentores de direitos autorais, é necessário compilar o OpenCV com a opção `OPENCV_ENABLE_NONFREE` habilitada.

Este tutorial não utiliza ambiente virtual Python com o `virtualenv` ou `conda`. Se você deseja utilizar o ambiente virtual veja tutorial [Using CMake to build and install OpenCV for Python and C++ in Ubuntu 20.04](https://rodosingh.medium.com/using-cmake-to-build-and-install-opencv-for-python-and-c-in-ubuntu-20-04-6c5881eebd9a).

Caso queira instalar diretamente no sistema, sem utilizar ambiente virtual, siga os passos abaixo.

## 1. Remova qualquer versão do OpenCV que você tenha instalado no seu sistema

```bash
pip uninstall opencv-python
pip uninstall opencv-contrib-python # Se você tiver instalado
sudo apt-get purge libopencv*
```

## 2. Prepare o ambiente para compilar o OpenCV

```bash
sudo apt-get update
sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
```

## 3. Clone os repositórios

Crie um diretório para o OpenCV e o opencv_contrib:

```bash
mkdir ~/opencv_build && cd ~/opencv_build # Escolha um diretório de sua preferência
```

### 3.1. Clone o repositório do OpenCV

Clone o repositório do OpenCV e faça o checkout para a tag da versão que você deseja compilar. Neste exemplo, vamos compilar a versão 4.10.0:

```bash
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.10.0
cd ~/opencv_build # Volte para o diretório anterior
```

### 3.2. Clone o repositório do opencv_contrib

Clone o repositório do opencv_contrib, que contém os módulos do OpenCV que não são distribuídos por padrão, como o `xfeatures2d` que contém os algoritmos SURF e BRIEF:

```bash
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.10.0
cd ~/opencv_build # Volte para o diretório anterior
```

## 4. Compile o OpenCV

Agora, é hora de compilar o OpenCV com o suporte ao opencv_contrib. Sigas os passos abaixo:

```bash
cd opencv
mkdir build && cd build
```

### 4.1. Configure o OpenCV

Para configurar o OpenCV, execute o comando abaixo. Note que a opção `OPENCV_ENABLE_NONFREE` está habilitada e o caminho para o diretório do opencv_contrib foi passado como argumento para a opção `OPENCV_EXTRA_MODULES_PATH`.

> [!IMPORTANT]
> Garanta que o binário do Python que você está utilizando é o mesmo que você deseja utilizar com o OpenCV. Estou utilizando o link simbólico `python` para o `python3` na versão 3.10.12. Se você não tem um link simbólico para o `python3`, você pode criar um link simbólico para o `python3` com o comando `sudo ln -s $(which python3) /usr/bin/python`. Ou apenas substitua `python` por `python3` no comando abaixo.

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_TEST=OFF \
-D BUILD_PERF_TESTS=OFF \
-D PYTHON3_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
-D WITH_V4L=ON \
-D WITH_EIGEN=ON \
-D WITH_TBB=ON \
-D WITH_IPP=ON \
-D WITH_OPENCL=ON \
-D WITH_OPENGL=ON \
-D BUILD_TIFF=ON \
-D PYTHON3_EXECUTABLE=$(which python) \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D INSTALL_C_EXAMPLES=ON \
-D PYTHON3_PACKAGES_PATH=$(python -c "import sysconfig; print(sysconfig.get_path('purelib'))") \
-D BUILD_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D PYTHON_DEFAULT_EXECUTABLE=$(which python) ..
```

Copie, cole e execute o comando acima para configurar a compilação do OpenCV. O argumento `..` no final do comando acima é necessário para que o cmake possa encontrar o arquivo `CMakeLists.txt` principal presente no diretório anterior (`~/opencv_build/opencv/`).

### 4.2. Compile e instale o OpenCV

Agora, compile o OpenCV com o comando abaixo:

```bash
make -j$(nproc) # Compila o OpenCV utilizando todos os núcleos do processador
```

Essa etapa pode demorar um pouco, dependendo do seu hardware. Após a compilação, instale o OpenCV com o comando abaixo:

```bash
sudo make install # Instala o OpenCV no sistema
```

Após a compilação e instalação, atualize os links simbólicos e o cache de bibliotecas compartilhadas com o comando abaixo:

```bash
sudo ldconfig
```

## 5. Verifique se o OpenCV foi instalado corretamente

Caso a compilação e instalação não tenham apresentado erros, você pode verificar se o OpenCV foi instalado corretamente para C++ e Python com os comandos abaixo:

### C++

```bash
pkg-config --modversion opencv4 # Saída esperada: 4.10.0
```

### Python

```bash
python -c "import cv2; print(cv2.__version__)" # Saída esperada: 4.10.0
```

Se você obteve a saída esperada para ambos os comandos, o OpenCV foi compilado e instalado corretamente no seu sistema. Agora você pode utilizar o módulo `xfeatures2d` para utilizar os algoritmos SURF e BRIEF.

## Referências

[Using CMake to build and install OpenCV for Python and C++ in Ubuntu 20.04](https://rodosingh.medium.com/using-cmake-to-build-and-install-opencv-for-python-and-c-in-ubuntu-20-04-6c5881eebd9a)

[Opencv Contrib ………](https://medium.com/@smilesajid14/opencv-contrib-17ff13e68ae3)

[How can I "Set OPENCV_ENABLE_NONFREE CMake option and rebuild the library in function 'create'"?](https://answers.opencv.org/question/201938/how-can-i-set-opencv_enable_nonfree-cmake-option-and-rebuild-the-library-in-function-create/)

[How to set OPENCV_ENABLE_NONFREE option?](https://stackoverflow.com/questions/66110067/how-to-set-opencv-enable-nonfree-option)
