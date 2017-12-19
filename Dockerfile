FROM ubuntu:latest

RUN apt-get update && apt-get -y install python3
RUN apt-get -y install curl
RUN cd /root/ && curl https://bootstrap.pypa.io/get-pip.py > get-pip.py && python3 get-pip.py
RUN pip install tensorflow
RUN apt-get -y install vim
COPY data /root/data