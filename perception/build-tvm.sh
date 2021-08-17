#!/bin/bash
WORKING_DIR=$(dirname $0)
TVM_EXPORT_SCRIPT="/etc/profile.d/utvm.sh"

cp "$WORKING_DIR/resources/tvm.cmake" "$WORKING_DIR/tvm/config.cmake"
pushd $(pwd)
cd $WORKING_DIR/tvm
make -j$(nproc) runtime
popd

#echo "# Setup TVM for the perceptive pipeline" > $TVM_EXPORT_SCRIPT
#echo "export PYTHONPATH=$PYTHONPATH:$WORKING_DIR/tvm/python" >> $TVM_EXPORT_SCRIPT
