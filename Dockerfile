# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3 AS mime-base
ENV PROGHOME=/mime
WORKDIR ${PROGHOME}
COPY . ${PROGHOME}/

FROM mime-base AS mime-capture
RUN apt update && apt install -y libclang-common-9-dev
RUN bash ${PROGHOME}/perception/build-tvm.sh
