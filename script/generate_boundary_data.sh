#!/usr/bin/env bash
for((m=1;m<=5;m++));
do
for((i=0;i<9;i++));
do
k=${i}+1
for((j=k;j<10;j++));
do
python ../src/find_boundary_data.py --model_path ../model/lenet-5.h5 --first_label_path ../data/original_data/class_${i}.npz --second_label_path ../data/original_data/class_${j}.npz --coefficient 0.00 --generate_num 1000 --save_path "../data/boundary_data/data_${i}&${j}/data_${i}&${j}_1000_part${m}.npz"
done
done
done
