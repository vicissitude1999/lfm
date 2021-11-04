rm -rf search-*
rm -rf eval-*
kubectl cp ../pdarts-LFM-F3/ ecepxie/renyi-login:/renyi-volume/LFM/
kubectl create -f train_search_lfm_f3.yaml