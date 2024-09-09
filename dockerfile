FROM tensorflow/tensorflow:latest-gpu

ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -r -g $GROUP_ID myuser && useradd -r -u $USER_ID -g myuser -m -d /home/myuser myuser
ENV SHELL /bin/bash

RUN mkdir -p /home/myuser/code && chown -R myuser:myuser /home/myuser/code

WORKDIR /home/myuser/code

RUN apt update
RUN pip install --upgrade pip
RUN pip install numpy==1.23.5
RUN pip install pandas==2.0.3
RUN pip install scikit-learn==1.2.2
RUN pip install matplotlib==3.3.4
RUN pip install aeon