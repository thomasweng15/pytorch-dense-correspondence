#!/bin/bash
FILES=/home/priya/code/data_volume/pdc_synthetic_2/logs_proto/*
for f in $FILES; do
	filename=$(basename "$f")
	touch $filename_only.yaml
	echo -e "logs_root_path: code/data_volume/pdc_synthetic_2/logs_proto" > ${filename}_only.yaml
	echo -e "" >> ${filename}_only.yaml
	echo -e "single_object_scenes_config_files:" >> ${filename}_only.yaml
	echo -e "- ${filename}.yaml" >> ${filename}_only.yaml
	echo -e "" >> ${filename}_only.yaml
	echo -e "multi_object_scenes_config_files: []" >> ${filename}_only.yaml
done
mv ./*.yaml /home/priya/code/config/dense_correspondence/dataset/composite/
