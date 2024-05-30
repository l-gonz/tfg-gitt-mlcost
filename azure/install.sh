#!/bin/bash

# Installation
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.12
sudo apt install -y python3.12-venv

ghtoken=$(az keyvault secret show -n Github-Token --vault-name kv-mlcost --query "value")
git clone https://github.com/l-gonz/tfg-gitt-mlcost.git --branch=azure tfg-gitt-mlcost-azure
cd tfg-gitt-mlcost-azure
git remote set-url origin https://azure-machine:${ghtoken}@github.com/l-gonz/tfg-gitt-mlcost.git
git config --local user.email "azure.machine@email.com"
git config --local user.name "azure-machine"

python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -r requirements.txt
python3.12 -m pip install -e .


# Commit
function commit() {
    git add azure/*
    git commit -m "Measure $1 in Azure machine" -m "$(cat /proc/cpuinfo  | grep 'name'| uniq)
$(cat /proc/meminfo | grep 'MemTotal')"
    git push origin
}

# Models
timestamp=$(date +%F_%T)
#python3.12 -m mlcost measure --openml -d electricity --log > "azure/electricity${timestamp}.log"
#mv output.csv azure/output${timestamp}.csv
commit electricity

python3.12 -m mlcost measure --openml -d covertype --log -m Linear > "azure/covertype${timestamp}.log"
python3.12 -m mlcost measure --openml -d covertype --log -m Forest >> "azure/covertype${timestamp}.log"
python3.12 -m mlcost measure --openml -d covertype --log -m SupportVector >> "azure/covertype${timestamp}.log"
python3.12 -m mlcost measure --openml -d covertype --log -m Neighbors >> "azure/covertype${timestamp}.log"
python3.12 -m mlcost measure --openml -d covertype --log -m NaiveBayes >> "azure/covertype${timestamp}.log"
python3.12 -m mlcost measure --openml -d covertype --log -m GradientBoost >> "azure/covertype${timestamp}.log"
python3.12 -m mlcost measure --openml -d covertype --log -m Neural >> "azure/covertype${timestamp}.log"
cat output.csv >> azure/output${timestamp}.csv
commit covertype

python3.12 -m mlcost measure --openml -d poker-hand --log -m Linear > "azure/poker${timestamp}.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m Forest >> "azure/poker${timestamp}.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m SupportVector >> "azure/poker${timestamp}.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m Neighbors >> "azure/poker${timestamp}.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m NaiveBayes >> "azure/poker${timestamp}.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m GradientBoost >> "azure/poker${timestamp}.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m Neural >> "azure/poker${timestamp}.log"
cat output.csv >> azure/output${timestamp}.csv
commit poker-hand
