#!/bin/bash

GPU_NUMBER=$1

docker run --rm -it --name="$(whoami)${GPU_NUMBER}" \
	--gpus "device=${GPU_NUMBER}" \
	-p 15000:80 \
	xl8/laser-gpu:latest python app.py
