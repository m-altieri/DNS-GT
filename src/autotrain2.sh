PT_EPOCHS=50
FT_EPOCHS=100
INC=10

for ((pt_epoch=$INC; pt_epoch<=PT_EPOCHS; pt_epoch+=$INC)); do
	(
		python3 train.py DNS-GT -r r2-"$pt_epoch"e --epochs $INC
		
		python3 train.py DNS-GT -r r2-"$pt_epoch"e-ft10e --start-from r2-"$pt_epoch"e --ft --from-pt -l b --epochs $INC
		python3 train.py DNS-GT -r r2-"$pt_epoch"e-ft10e --evaluate --eval-data train &
		
		for ((ft_epoch=$INC; ft_epoch<$((FT_EPOCHS)); ft_epoch+=$INC)); do

			python3 train.py DNS-GT -r r2-"$pt_epoch"e-ft"$((ft_epoch+INC))"e --start-from r2-"$pt_epoch"e-ft"$ft_epoch"e --ft -l b --epochs $INC
			python3 train.py DNS-GT -r r2-"$pt_epoch"e-ft"$((ft_epoch+INC))"e --ft -l b --evaluate --eval-data train &
		done
	) &
	
	sleep 30
done

wait

echo "**********   ALL TRAININGS HAVE FINISHED   **********"


python3 -c "from utils.evaluation import Evaluation; evaluation = Evaluation(); evaluation.collect_results(verbose=True)"

echo "********** RESULTS COLLECTED **********"
