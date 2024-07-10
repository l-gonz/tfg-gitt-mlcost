#!/bin/bash

timestamp=$(date +%F_%T)
python -m mlcost measure --log -cv 5 --openml -d electricity
python -m mlcost measure --log -cv 5 --openml -d electricity --parallel

mv output.csv out/output_parallel_${timestamp}.csv

git add *
git commit -m "Measure parallel run in Azure machine" -m "$(cat /proc/cpuinfo  | grep 'name'| uniq)
$(cat /proc/meminfo | grep 'MemTotal')"
git push origin