#!/bin/sh

while true
do
  kubectl delete pod renyi-login
  kubectl create -f pod_login.yaml
  echo created
  sleep 21600
done