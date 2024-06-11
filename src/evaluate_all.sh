sleep 5 ; python3 train.py DNS-GT -r run3-focal-loss-30e-ft50e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r run2-50e-ft25e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r run3-focal-loss-30e-ft25e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r run2-30e-ft50e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r run2-30e-ft25e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft100e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft45e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft35e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft25e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft20e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft15e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r b-run1-15e-ft10e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r run1-15e-ft10e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py DNS-GT -r run1-15e-ft5e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py W2V --type SkipGram -r run1-15e-ft50e --evaluate -l b --eval-data train &
sleep 5 ; python3 train.py W2V --type CBOW -r run1-15e-ft50e --evaluate -l b --eval-data train &
