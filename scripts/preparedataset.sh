#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
wget -P ../../ http://assets.laboro.ai/laborotomato/laboro_tomato.zip
cd ../../
unzip laboro_tomato.zip
cp YOLOv5-Lite/scripts/convert_json2yolo.py laboro_tomato
mkdir -p laboro_tomato/images/
cp -r laboro_tomato/train laboro_tomato/images/train
cp -r laboro_tomato/test laboro_tomato/images/test
python laboro_tomato/convert_json2yolo.py
