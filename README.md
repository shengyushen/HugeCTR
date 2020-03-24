nvidia-docker run -it --ipc=host -v /root/ssy:/root/ssy --name ssyHugerctr nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
nvidia-docker start ssyHugerctr
nvidia-docker exec -it ssyHugerctr /bin/bash

#in docker
apt update -o Acquire::https::developer.download.nvidia.com::Verify-Peer=false

apt install -y vim cmake

# make
cmake  -DCMAKE_BUILD_TYPE=Release -DSM=70


# HugeCTR #
HugeCTR is a high-efficiency GPU framework designed for Click-Through-Rate (CTR) estimation training.

Design Goals:
* Optimized for recommender system
* Easy to be customized

Please find more introductions in our [**HugeCTR User Guide**](docs/hugectr_user_guide.md) and doxygen files in directory `docs\`

## Requirements ##
* cuBLAS >= 9.1
* Compute Capability >= 60 (P100)
* CMake >= 3.8
* cuDNN >= 7.5
* NCCL >= 2.0
* Clang-Format 3.8
* OpenMPI >= 4.0 (optional, if require multi-nodes training)

## Build ##
### Init Git ###
```shell
$ git submodule init
$ git submodule update
```

### Build with Release ###
Compute Capability can be specified by `-DSM=XX`, which is SM=60 by default. Only one Compute Capability is avaliable to be set.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM=XX ..
$ make
```

### Build with Debug ###
Compute Capability can be specified by `-DSM=XX`, which is SM=60 by default. Only one Compute Capability is avaliable to be set.
```shell
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug -DSM=XX ..
$ make
```
### Build with Mixed Precision (WMMA) Support ###
To use mixed precision training, enable USE_WMMA and set SCALER to 128/256/512/1024 by:
```shell
$ mkdir -p build
$ cd build
$ cmake -DSM=XX -DUSE_WMMA=ON -DSCALER=YYY ..
```

## Run ##
Please refer to samples/*

## Coding Style and Refactor ##
Default coding style follows Google C++ coding style [(link)](https://google.github.io/styleguide/cppguide.html).
This project also uses `Clang-Format`[(link)](https://clang.llvm.org/docs/ClangFormat.html) to help developers to fix style issue, such as indent, number of spaces to tab, etc.
The Clang-Format is a tool that can auto-refactor source code.
Use following instructions to install and enable Clang-Format:
### Install ###
```shell
$ sudo apt-get install clang-format
```
### Run ###
```shell
# First, configure Cmake as usual 
$ mkdir -p build
$ cd build
$ cmake -DCLANGFORMAT=ON ..
# Second, run Clang-Format
$ cmake --build . --target clangformat
# Third, check what Clang-Format did modify
$ git status
# or
$ git diff
```

## Document Generation ##
Doxygen is supported in HugeCTR and by default on-line documentation browser (in HTML) and an off-line reference manual (in LaTeX) can be generated within `docs/`.
### Install ###
[Download doxygen](http://www.doxygen.nl/download.html)
### Generation ###
Within project `home` directory
```shell
$ doxygen
```

## File Format ##
Totally three kinds of files will be used as input of HugeCTR Training: configuration file (.json), model file, data set.

### Configuration File ###
Configuration file should be a json format file e.g. [simple_sparse_embedding.json](utest/session/simple_sparse_embedding.json)

There are four sessions in a configuration file: "solver", "optimizer", "data", "layers". The sequence of these sessions is not restricted.
* You can specify the device (or devices), batchsize, model_file.. in `solver` session;
* and the `optimizer` that will be used in every layer.
* File list and data set related configurations will be specified under `data` session.
* Finally, layers should be listed under `layers`. Note that embedders should always be the first layer.

### Model File ###
Model file is a binary file that will be loaded for weight initilization.
In model file weight will be stored in the order of layers in configuration file.

### Data Set ###
A data set includes a ASCII format file list and a set of data in binary format.

A file list starts with a number which indicate the number of files in the file list, then comes with the path of each data file.
```shell
$ cat simple_sparse_embedding_file_list.txt
10
./simple_sparse_embedding/simple_sparse_embedding0.data
./simple_sparse_embedding/simple_sparse_embedding1.data
./simple_sparse_embedding/simple_sparse_embedding2.data
./simple_sparse_embedding/simple_sparse_embedding3.data
./simple_sparse_embedding/simple_sparse_embedding4.data
./simple_sparse_embedding/simple_sparse_embedding5.data
./simple_sparse_embedding/simple_sparse_embedding6.data
./simple_sparse_embedding/simple_sparse_embedding7.data
./simple_sparse_embedding/simple_sparse_embedding8.data
./simple_sparse_embedding/simple_sparse_embedding9.data
```

A data file (binary) contains a header and data (many samples). 

Header Definition:
```c
typedef struct DataSetHeader_{
  long long number_of_records; //the number of samples in this data file
  long long label_dim; //dimension of label
  long long slot_num; //the number of slots in each sample 
  long long reserved; //reserved for future use
} DataSetHeader;
```

Data Definition (each sample):
```c
typedef struct Data_{
  int label[label_dim];
  Slot slots[slot_num];
} Data;

typedef struct Slot_{
  int nnz;
  T*  keys; //long long or uint
} Slot;
```
