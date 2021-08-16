#!/bin/sh



echo "prepare for training------."

python3 ./src/prepare.py

echo "prepare feature of lgb for training------."

python3 ./src/prepareFeatureLightgbm.py

echo "prepare for deepfm------."

python3 ./src/prepare_data_for_deepfm.py

echo "training deep_model1-----"

python3 ./src/my_deep_v2_v1.py

echo "training deep_model2-----"

python3 ./src/my_deep_v2_v2.py

echo "training lightgbm-----"

python3 ./src/trainLightgbm.py

echo "training deepfm-----"

python3 ./src/DeepFM.py

echo "training AutoInt-----"

python3 ./src/AutoInt.py

echo "inference...."

python3 ./src/inference.py


echo "finish!"
