# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/l4t-ml:r32.6.1-py3 AS mime-base
ENV PROGHOME=/mime
WORKDIR ${PROGHOME}
COPY . ${PROGHOME}/

FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-base AS tvm-builder
RUN apt update && apt install -y build-essential cmake llvm-9-dev libclang-9-dev
ENV BUILDHOME=/bld
WORKDIR ${BUILDHOME}
COPY . ${BUILDHOME}/
RUN bash ${BUILDHOME}/perception/build-tvm.sh

FROM mime-base AS mime-capture
RUN rm -rf ${PROGHOME}/perception/tvm
COPY --from=tvm-builder ${BUILDHOME}/perception/tvm ${PROGHOME}/perception/tvm
