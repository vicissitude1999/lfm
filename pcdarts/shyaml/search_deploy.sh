cd ..
rm -rf search-*
rm -rf eval-*
kubectl cp ../pcdarts/ ecepxie/renyi-login:/renyi-volume/LFM/
kubectl create -f search.yaml