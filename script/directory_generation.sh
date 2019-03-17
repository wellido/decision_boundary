#!/usr/bin/env bash
for((i=0;i<9;i++));
do
k=${i}+1
for((j=k;j<10;j++))
do
mkdir "../data/boundary_data/data_${i}&${j}"
done
done