#!/bin/bash

# Run to get this script
# git clone https://github.com/l-gonz/tfg-gitt-mlcost.git --branch=azure tfg-gitt-mlcost-azure

# Installation
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
az login

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install -y python3.12
sudo apt install -y python3.12-venv

ghtoken=$(az keyvault secret show -n Github-Token --vault-name kv-mlcost --query "value" -o tsv)
cd ~/tfg-gitt-mlcost
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

function moveOutput() {
    if [ -f $1 ]; then
        echo >> $1
        tail -n +2 output.csv >> $1
        rm output.csv
    else
        mv output.csv $1
    fi
}

# Models
timestamp=$(date +%F_%T)
python3.12 -m mlcost measure --openml -d electricity --log > "azure/${timestamp}_electricity.log"
moveOutput azure/output${timestamp}.csv
commit electricity

python3.12 -m mlcost measure --openml -d covertype --log -m Linear > "azure/${timestamp}_covertype.log"
python3.12 -m mlcost measure --openml -d covertype --log -m Forest >> "azure/${timestamp}_covertype.log"
python3.12 -m mlcost measure --openml -d covertype --log -m Neighbors >> "azure/${timestamp}_covertype.log"
python3.12 -m mlcost measure --openml -d covertype --log -m NaiveBayes >> "azure/${timestamp}_covertype.log"
python3.12 -m mlcost measure --openml -d covertype --log -m GradientBoost >> "azure/${timestamp}_covertype.log"
moveOutput azure/output${timestamp}.csv
commit covertype

python3.12 -m mlcost measure --openml -d poker-hand --log -m Linear > "azure/${timestamp}_poker.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m Forest >> "azure/${timestamp}_poker.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m Neighbors >> "azure/${timestamp}_poker.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m NaiveBayes >> "azure/${timestamp}_poker.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m GradientBoost >> "azure/${timestamp}_poker.log"
moveOutput azure/output${timestamp}.csv
commit poker-hand

# Overly long models
python3.12 -m mlcost measure --openml -d covertype --log -m Neural >> "azure/${timestamp}_covertype.log"
python3.12 -m mlcost measure --openml -d poker-hand --log -m Neural >> "azure/${timestamp}_poker.log"
moveOutput azure/output${timestamp}.csv
commit neural

python3.12 -m mlcost measure --openml -d poker-hand --log -m SupportVector >> "azure/${timestamp}_poker.log"
python3.12 -m mlcost measure --openml -d covertype --log -m SupportVector >> "azure/${timestamp}_covertype.log"
moveOutput azure/output${timestamp}.csv
commit support-vector
