# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3 AS mime-base
ENV PROGHOME=/mime
WORKDIR ${PROGHOME}

FROM mime-base AS mime-capture
COPY perception/. ${PROGHOME}/
