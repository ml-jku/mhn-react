FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

WORKDIR /mhn-react
COPY . /workdir/mhn-react

RUN /opt/conda/bin/conda env create -f /workdir/mhn-react/tools/docker/env.yml && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/envs/mhnreact_env/bin/python -m pip install /workdir/mhn-react
RUN /opt/conda/envs/mhnreact_env/bin/python -m pip install /workdir/mhn-react/data/temprel-fortunato/template-relevance-master/.

ENV PATH /opt/conda/envs/mhnreact_env/bin:$PATH
