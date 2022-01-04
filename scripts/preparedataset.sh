#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
wget -P ../../ http://assets.laboro.ai/laborotomato/laboro_tomato.zip
cd ../../
unzip -q laboro_tomato.zip
cp YOLOv5-Lite/scripts/preparedataset.py laboro_tomato
python laboro_tomato/preparedataset.py
