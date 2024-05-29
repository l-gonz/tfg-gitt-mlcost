#!/bin/bash

# Installation
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.12
sudo apt install -y python3.12-venv

cd ~
git clone https://azure-machine:${github-token}@github.com/l-gonz/tfg-gitt-mlcost.git --branch=azure azure-test
cd tfg-gitt-mlcost
git checkout azure
git config --local user.email "placeholder@email.com"
git config --local user.name "azure-machine"

python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -r requirements.txt
python3.12 -m pip install -e .


# Commit
function commit() {
    mv output.csv azure/output${timestamp}.csv
    git add azure/*
    cpu=$(cat /proc/cpuinfo  | grep 'name'| uniq)
    mem=
    git commit -m "Measure covertype in Azure machine" -m "$(cat /proc/cpuinfo  | grep 'name'| uniq)
    $(cat /proc/meminfo | grep 'MemTotal')"
}

# Models
timestamp=$(date +%F_%T)
python3.12 -m mlcost measure --log -m Linear > "azure/iris${timestamp}.log"
python3.12 -m mlcost measure --log -m Forest >> "azure/iris${timestamp}.log"
commit

# python3.12 -m mlcost measure --openml -d covertype --log -m Linear > "azure/covertype${timestamp}.log"
# python3.12 -m mlcost measure --openml -d covertype --log -m Forest >> "azure/covertype${timestamp}.log"
# python3.12 -m mlcost measure --openml -d covertype --log -m SupportVector >> "azure/covertype${timestamp}.log"
# python3.12 -m mlcost measure --openml -d covertype --log -m Neighbors >> "azure/covertype${timestamp}.log"
# python3.12 -m mlcost measure --openml -d covertype --log -m NaiveBayes >> "azure/covertype${timestamp}.log"
# python3.12 -m mlcost measure --openml -d covertype --log -m GradientBoost >> "azure/covertype${timestamp}.log"
# python3.12 -m mlcost measure --openml -d covertype --log -m Neural >> "azure/covertype${timestamp}.log"
# commit

# python3.12 -m mlcost measure --openml -d poker-hand --log -m Linear > "azure/poker${timestamp}.log"
# python3.12 -m mlcost measure --openml -d poker-hand --log -m Forest >> "azure/poker${timestamp}.log"
# python3.12 -m mlcost measure --openml -d poker-hand --log -m SupportVector >> "azure/poker${timestamp}.log"
# python3.12 -m mlcost measure --openml -d poker-hand --log -m Neighbors >> "azure/poker${timestamp}.log"
# python3.12 -m mlcost measure --openml -d poker-hand --log -m NaiveBayes >> "azure/poker${timestamp}.log"
# python3.12 -m mlcost measure --openml -d poker-hand --log -m GradientBoost >> "azure/poker${timestamp}.log"
# python3.12 -m mlcost measure --openml -d poker-hand --log -m Neural >> "azure/poker${timestamp}.log"
# commit
