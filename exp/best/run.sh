#!/bin/bash
for i in $(seq 1 20);
do
	python3 all.py german $i > ./log/$i.log
done
