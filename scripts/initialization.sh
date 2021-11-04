cp ~/Documents/config ~/.kube/
rm ~/Documents/config
kubectl delete pod renyi-login
kubectl create -f login.yaml
