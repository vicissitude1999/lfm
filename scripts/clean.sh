for model in darts darts-lfm pcdarts pcdarts-lfm; do
rm -rf "$model/trash"

for d in $model/search-*; do
if (test -e "$d/log.txt") && (cat "$d/log.txt" | grep "epoch 49")
then
	mv $d runs/$model
else
	mv $d runs_trash/$model
fi
done

for d in $model/eval-*; do
if (test -e "$d/log.txt") && (cat "$d/log.txt" | grep "epoch 599")
then
	mv $d runs/$model
else
	mv $d runs_trash/$model
fi
done

done


for model in pdarts pdarts-lfm; do
rm -rf "$model/trash"

for d in $model/search-*; do
if (test -e "$d/log.txt") && (cat "$d/log.txt" | grep "Total searching time")
then
	mv $d runs/$model
else
	mv $d runs_trash/$model
fi
done

for d in $model/eval-*; do
if (test -e "$d/log.txt") && (cat "$d/log.txt" | grep "epoch 599")
then
	mv $d runs/$model
else
	mv $d runs_trash/$model
fi
done

done