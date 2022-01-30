#!/bin/bash

for i in {1..10}
do
    python src/main.py "${@:1}"
done

stress -c 2 -i 2 -m 2 &
sleep 60
for i in {1..10}
do
    python src/main.py "${@:1}"
done
ps -ef | awk '/stress/{print $2}' | xargs kill

stress -c 6 -i 6 -m 6 -d 6 &
sleep 60
for i in {1..10}
do
    python src/main.py "${@:1}"
done
ps -ef | awk '/stress/{print $2}' | xargs kill