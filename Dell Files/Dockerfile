#use ubuntu 22.04 as base image must have
FROM ubuntu:22.04

#Set non-interactive mode and disable .pyc files / buffering
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYCODE=1
ENV PYTHONUNBUFFERED=1

#install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        python3-tk \
        build-essential \
        liblapack-dev \
        libatlas-base-dev \
        gcc \
        curl \
        git \
        graphviz \
        libgraphviz-dev \
        libgli-mesa-glx \
        pkg-config \
        lsb-release \
        bsdmainutils \ 
        # Gstreaemer not needed, just in case
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 && \
    rm -rf /var/lib/apt/lists/*
    
