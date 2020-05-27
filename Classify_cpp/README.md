vi CMakeLists.txt & modify path for opencv, libtorch

mkdir build

cd build

cmake ..

make

Classify_cpp


Experiment result:

Resnet50 model need less than 15 seconds for computing one image 1000 times with C++ API, while 18 seconds needed  in Python. Running in C++, we can save 16% time consume.
