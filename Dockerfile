FROM eywalker/attorch


RUN apt-get update -y \
    && apt-get install -y graphviz graphviz-dev \ 
    && pip3 install graphviz \
    && pip3 install gpustat

RUN pip3 install jupyter
RUN pip3 install git+https://github.com/datajoint/datajoint-python.git

ADD . /src/blinkende_lichter
RUN pip3 install -e /src/blinkende_lichter

WORKDIR /notebooks


