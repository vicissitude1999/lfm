cd ..
rm -rf search-*
rm -rf eval-*
kubectl cp ../pcdarts-lfm/ ecepxie/renyi-login:/renyi-volume/LFM/

cd shyaml
kubectl create -f search.yaml