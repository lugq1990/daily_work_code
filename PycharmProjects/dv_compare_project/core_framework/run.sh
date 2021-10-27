#!/bin/bash

for i in "$@"; do
  IFS='=' read -ra ADDR <<< "$i"
  printf '%s\n' "${ADDR[0]}"
  printf '%s\n' "${ADDR[1]}"
  sed -i "s!#{${ADDR[0]}}#!${ADDR[1]}!gI" config.py
done

python3 init_test_case.py
python3 robot_auto_test.py --dryrun   devops_testcases
python3 robot_auto_test.py --listener ud_listener/listener001.py devops_testcases
# python3 /app/robot_auto_test.py /app/devops_testcases/test_case.robot
