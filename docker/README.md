# tvm_study
## build
1. build with cuda support
```
git clone --recursive https://github.com/apache/tvm
# install pre-requisties
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
mkdir build
cp cmake/config.cmake build
cd build
# edit build/config.cmake to turn on CUDA support
# set(USE_CUDA ON)
cmake ..
make -j12

# you will get the built libtvm.so and libtvm_runtime.so library after previous step

# optional: for virtual environment 
apt-get install -y python3.10-venv
python -m venv tvm
# install pip3 for installing python package dependencies
apt-get install -y python3-pip 

# add tvm python package to python path
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
