for ((i=1; i<=100; i++))
do
	python3 mlp9.py german > log/mlp9_choose_seed_$i.log
done
