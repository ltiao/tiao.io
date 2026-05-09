---
aliases:
  - /post/docker-image-for-machine-learning-research-and-development-pytorch-jax-tensorflow/
  # Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "A Docker Image for Machine Learning Research and Development in PyTorch, JAX and TensorFlow"
subtitle: ""
summary: ""
authors: []
tags:
- Docker
- PyTorch
- TensorFlow
- JAX
categories: []
date: 2021-03-03T16:28:47+01:00
lastmod: 2021-03-03T16:28:47+01:00
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

My Dockerfiles: https://github.com/ltiao/dockerfiles

Derived from the official TensorFlow Docker images, in particular the GPU 
flavor of the image based on the NVIDIA CUDA Docker image.
The sources are available at: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/dockerfiles

If you are just interested in running TensorFlow with GPU support and Jupyter 
notebooks installed, just use the image/tag: `tensorflow/tensorflow:2.4.0-gpu-jupyter`:

```dockerfile
FROM tensorflow/tensorflow:2.4.0-gpu-jupyter
```

This image installs both PyTorch and JAX and allows one finer-grain control 
over which minor version of Python 3 to use. For example, 3.9 is the latest 
version, 3.6 is the default shipped with Debian, and some Python libraries I
have used strictly require version 3.7.

I created this out of frustration with only being able to find a dedicated 
TensorFlow image, or a dedicated Torch image, and trying to "append" Tensorflow
to the Torch image, or vice versa caused major headaches. Other times I have
appended Torch to the TensorFlow image to take advantage of the fact that it
is based on CUDA and already has such dependencies installed, only to find that
it is only operable with a specific version of Python different from the version
that associated with the TensorFlow installation in the image.

In the end I found it best to have control over the base layer of the image, 
namely the CUDA installation, the OS version, CUDA version, and the incrementally build up from this layer:
the Python installation with its particular version, installing TensorFlow with
GPU/CUDA support associated with this version, similarly a PyTorch installation
that is associated with the same Python version and takes advantage of the GPU/CUDA support.
similarly with JAX, and then other tools for data analysis and visualization,
such as Pandas, Matplotlib and Seaborn and finally the interactive computing tools,
i.e. IPython, Jupyter and its descendents.

The image takes advantage of build arguments, so it is easy to build variants of
this image with differing combinations of UBUNTU, CUDA, PYTHON, TENSORFLOW, 
TORCH, and JAX versions.

## Running the Image 

Pulls from Docker Hub so you and pull and run right away (just running is 
enough as it will automatically be pulled for you if it detects that the image 
is not available in your local Docker repository)

### Run with GPU Support using NVIDIA Docker Container Runtime

## Building the Image from sources


## Interactive computing

IPython, Jupyter Notebooks and Jupyter Lab

## Experiment scripts

### Publication-quality plots

---

The full Dockerfile is produced below:

```dockerfile
ARG UBUNTU_VERSION=18.04
ARG ARCH=
ARG CUDA=11.0

FROM nvidia/cuda${ARCH:+-$ARCH}:${CUDA}-base-ubuntu${UBUNTU_VERSION} as nvidia
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
ARG CUDNN=8.0.4.30-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.1.3-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libcudnn8=${CUDNN}+cuda${CUDA} \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip

# Install TensorRT if not building for PowerPC
RUN [[ "${ARCH}" = "ppc64le" ]] || { apt-get update && \
        apt-get install -y --no-install-recommends libnvinfer${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        libnvinfer-plugin${LIBNVINFER_MAJOR_VERSION}=${LIBNVINFER}+cuda${CUDA} \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*; }

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
    && ldconfig

FROM nvidia as python

# Python
# ------

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ARG PYTHON_VERSION=3.8
ARG PYTHON_MAJOR_VERSION=3
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_MAJOR_VERSION}-pip

RUN python${PYTHON_VERSION} -m pip --no-cache-dir install --upgrade \
    "pip<21.0.1" \
    setuptools

# Some TF tools expect a "python" binary
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}
RUN ln -s $(which python${PYTHON_MAJOR_VERSION}) /usr/local/bin/python

FROM python as common

# TensorFlow
# ----------

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow-gpu
ARG TF_PACKAGE_VERSION=2.4.1
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

# Torch
# -----
ARG TORCH_PACKAGE_VERSION=1.7.1+cu110
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
    torch${TORCH_PACKAGE_VERSION:+==${TORCH_PACKAGE_VERSION}} \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# JAX
# ---
ARG JAX_PACKAGE_VERSION=0.2.9
ARG JAXLIB_PACKAGE_VERSION=0.1.61+cuda110
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir \
    jax${JAX_PACKAGE_VERSION:+==${JAX_PACKAGE_VERSION}} \
    jaxlib${JAXLIB_PACKAGE_VERSION:+==${JAXLIB_PACKAGE_VERSION}} -f https://storage.googleapis.com/jax-releases/jax_releases.html

FROM common as base
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

FROM base as jupyter
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir jupyter matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
RUN python${PYTHON_VERSION} -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN mkdir /.local && chmod a+rwx /.local
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
EXPOSE 8888

RUN python${PYTHON_VERSION} -m ipykernel.kernelspec

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --ip 0.0.0.0 --no-browser --allow-root"]
```

