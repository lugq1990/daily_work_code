#!/bin/bash

for i in "$@"; do
  IFS='=' read -ra ADDR <<< "$i"
  printf '%s\n' "${ADDR[0]}"
  printf '%s\n' "${ADDR[1]}"
  sed -i "s!#{${ADDR[0]}}#!${ADDR[1]}!gI" config.py
done

pip install -r requirements.txt

python init_test_case.py

python robot_auto_test.py --dryrun   devops_testcases

