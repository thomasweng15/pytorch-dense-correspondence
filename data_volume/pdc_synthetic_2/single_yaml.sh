#!/bin/bash
FILES=/home/priya/code/data_volume/pdc_synthetic_2/logs_proto/*
for f in $FILES; do
	filename=$(basename "$f")
	touch $filename.yaml
	echo -e "logs_root_path: code/data_volume/pdc_synthetic_2/logs_proto" > ${filename}.yaml
	echo -e "object_id: \"${filename}\"" >> ${filename}.yaml
	echo -e "train:" >> ${filename}.yaml	
	echo -e "- \"${filename}\"" >> ${filename}.yaml
	echo -e "test:" >> ${filename}.yaml	
	echo -e "- \"${filename}_test\"" >> ${filename}.yaml
done
mv ./*.yaml /home/priya/code/config/dense_correspondence/dataset/single_object/ 
