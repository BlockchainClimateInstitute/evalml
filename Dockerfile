FROM FROM continuumio/miniconda3

COPY . .
ADD environment.yml environment.yml

RUN conda env create -f environment.yml
ENV PATH /opt/conda/envs/t5env/bin:$PATH
RUN /bin/bash -c "source activate t5env"
