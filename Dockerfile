FROM quay.io/jupyter/pytorch-notebook:cuda12-ubuntu-24.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/home/jovyan/work/src

WORKDIR /home/jovyan/work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

COPY notebooks/ /home/jovyan/work/notebooks/
COPY src/ /home/jovyan/work/src/
COPY scripts/ /home/jovyan/work/scripts/
COPY README.md /home/jovyan/work/README.md

RUN mkdir -p /home/jovyan/work/data \
    /home/jovyan/work/models \
    /home/jovyan/work/results

EXPOSE 8888

CMD ["start-notebook.sh", "--NotebookApp.token=''"]