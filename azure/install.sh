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
cd ~/tfg-gitt-mlcost-azure
git remote set-url origin https://azure-machine:${ghtoken}@github.com/l-gonz/tfg-gitt-mlcost.git
git config --local user.email "azure.machine@email.com"
git config --local user.name "azure-machine"

python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -r requirements.txt
python3.12 -m pip install -e .

#sh azure/big-data.sh
sh azure/parallel.sh
