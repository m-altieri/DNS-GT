PT_EPOCHS=('5e' '10e' '15e' '20e' '25e' '30e')

for e in "${PT_EPOCHS[@]}"; do
	echo "Starting block in background for '$e'"
	(
		python3 train.py DNS-GT -r "$e" --epochs 5
		cp -r ../runs/DNS-GT/pretrained/"$e" ../runs/DNS-GT/b/"$e"-ft5e
		python3 train.py DNS-GT -r "$e"-ft5e --ft -l b --epochs 5
		python3 train.py DNS-GT -r "$e"-ft5e --ft -l b --evaluate --eval-data train &
		cp -r ../runs/DNS-GT/b/"$e"-ft5e ../runs/DNS-GT/b/"$e"-ft10e
		python3 train.py DNS-GT -r "$e"-ft10e --ft -l b --epochs 5
		python3 train.py DNS-GT -r "$e"-ft10e --ft -l b --evaluate --eval-data train &
		cp -r ../runs/DNS-GT/b/"$e"-ft10e ../runs/DNS-GT/b/"$e"-ft15e
		python3 train.py DNS-GT -r "$e"-ft15e --ft -l b --epochs 5
		python3 train.py DNS-GT -r "$e"-ft15e --ft -l b --evaluate --eval-data train &
		cp -r ../runs/DNS-GT/b/"$e"-ft15e ../runs/DNS-GT/b/"$e"-ft20e
		python3 train.py DNS-GT -r "$e"-ft20e --ft -l b --epochs 5
		python3 train.py DNS-GT -r "$e"-ft20e --ft -l b --evaluate --eval-data train &
		cp -r ../runs/DNS-GT/b/"$e"-ft20e ../runs/DNS-GT/b/"$e"-ft25e
		python3 train.py DNS-GT -r "$e"-ft25e --ft -l b --epochs 5
		python3 train.py DNS-GT -r "$e"-ft25e --ft -l b --evaluate --eval-data train &
		cp -r ../runs/DNS-GT/b/"$e"-ft25e ../runs/DNS-GT/b/"$e"-ft30e
		python3 train.py DNS-GT -r "$e"-ft30e --ft -l b --epochs 5
		python3 train.py DNS-GT -r "$e"-ft30e --ft -l b --evaluate --eval-data train &
	) &
	sleep 10
done

wait

echo "**********   ALL TRAININGS HAVE FINISHED   **********"
