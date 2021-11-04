cd ..
rm -rf search-*
rm -rf eval-*
kubectl cp ../darts/ ecepxie/renyi-login:/renyi-volume/LFM/
kubectl create -f search.yaml