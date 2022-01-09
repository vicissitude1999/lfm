#cd ../scripts
#bash sync.sh

kubectl cp /home/renyi/Documents/LFM/shyaml ecepxie/renyi-login:/renyi-volume/LFM/
kubectl create -f train.yaml