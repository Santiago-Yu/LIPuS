#base image
FROM python:3.7.10
#Add source file
ADD . ./LIPuS
# set wordir
WORKDIR /LIPuS
# install requirements
RUN pip install -r requirements.txt
RUN cd /LIPuS/code2inv/graph_encoder; make clean; make
RUN cd /
WORKDIR /
