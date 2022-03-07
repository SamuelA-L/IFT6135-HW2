python run_exp_lstm.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer adam --exp_id 1
echo exp 1 done
python run_exp_lstm.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer adamw --exp_id 2
echo exp 2 done
python run_exp_lstm.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer sgd --exp_id 3
echo exp 3 done
python run_exp_lstm.py --model lstm --layers 1 --batch_size 16 --log --epochs 10 --optimizer momentum --exp_id 4
echo exp 4 done
python run_exp_lstm.py --model lstm --layers 2 --batch_size 16 --log --epochs 10 --optimizer adamw --exp_id 5
echo exp 5 done
python run_exp_lstm.py --model lstm --layers 4 --batch_size 16 --log --epochs 10 --optimizer adamw --exp_id 6
echo exp 6 done