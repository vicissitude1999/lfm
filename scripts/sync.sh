cd ..

for f in *; do
if [ $f != "runs" ] && [ $f != "ritvik" ]
then
  echo $f
  kubectl cp $f ecepxie/renyi-login:/renyi-volume/LFM/
fi
done