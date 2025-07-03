#!/bin/bash
# Активация системного OpenVINO
source /opt/intel/openvino/setupvars.sh

# Добавление Python-модулей OpenVINO в PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/opt/intel/openvino/python/python3.9:/opt/intel/openvino/tools