PT_EPOCHS=50
FT_EPOCHS=100
INC=10

# pretraining outer loop
for ((pt_epoch=$INC; pt_epoch<=PT_EPOCHS; pt_epoch+=$INC)); do

	# if it's the first iteration, start from scratch, otherwise --start-from the previous one
	if [[ "$pt_epoch" == $INC ]]; then
		python3 train.py DNS-GT -r r3-"$pt_epoch"e --epochs $INC
	else
		python3 train.py DNS-GT -r r3-"$pt_epoch"e --epochs $INC --start-from r3-"$((pt_epoch-INC))"e
	
	# after pretraining, run block in background to finetune on that pretraining and go on to pretrain with more epochs
	(
		for ((ft_epoch=$INC; ft_epoch<$((FT_EPOCHS)); ft_epoch+=$INC)); do
		
			# if it's the first iteration of finetuning, start from the pretrained run, otherwise start from the last finetuned run
			if [[ "$ft_epoch" == $INC ]]; then
				python3 train.py DNS-GT -r r3-"$pt_epoch"e-ft10e --start-from r3-"$pt_epoch"e --ft --from-pt -l b --epochs $INC
				python3 train.py DNS-GT -r r3-"$pt_epoch"e-ft10e --evaluate --eval-data train &
			else
				python3 train.py DNS-GT -r r3-"$pt_epoch"e-ft"$((ft_epoch+INC))"e --start-from r3-"$pt_epoch"e-ft"$ft_epoch"e --ft -l b --epochs $INC
				python3 train.py DNS-GT -r r3-"$pt_epoch"e-ft"$((ft_epoch+INC))"e --ft -l b --evaluate --eval-data train &
			fi
		done
	) &
	
done

wait

echo "**********   ALL TRAININGS HAVE FINISHED   **********"


python3 -c "from utils.evaluation import Evaluation; evaluation = Evaluation(); evaluation.collect_results(verbose=True)"

echo "********** RESULTS COLLECTED **********"
