FROM registry.gitlab.com/mefortunato/docker-images/rdkit:2019.03.3-tf-mpi

COPY requirements.txt requirements.txt

RUN apt-get install -y unzip git && \
    pip install -r requirements.txt && \
    pip install -e "git://github.com/connorcoley/rdchiral.git#egg=rdchiral"

COPY . /usr/local/temprel

RUN pip install -e /usr/local/temprel && \
    cp /usr/local/temprel/bin/* /usr/local/bin
