#!/bin/sh

cd data

wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d .
rm train2017.zip
mv train2017 image_dataset
