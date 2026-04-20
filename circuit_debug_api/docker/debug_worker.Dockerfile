ARG BASE_IMAGE=dustynv/l4t-pytorch:r36.4.0
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=
ENV PYTHONDONTWRITEBYTECODE=1

LABEL lab.debug_worker.base_image="${BASE_IMAGE}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenblas-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.debug_worker_container.txt /tmp/requirements.debug_worker_container.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /tmp/requirements.debug_worker_container.txt
