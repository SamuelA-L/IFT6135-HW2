python run_exp_vit.py --model vit --layers 2 --batch_size 128 --log --epochs 10 --optimizer adam --exp_id 7
echo exp 7 done
python run_exp_vit.py --model vit --layers 2 --batch_size 128 --log --epochs 10 --optimizer adamw --exp_id 8
echo exp 8 done
python run_exp_vit.py --model vit --layers 2 --batch_size 128 --log --epochs 10 --optimizer sgd --exp_id 9
echo exp 9 done
python run_exp_vit.py --model vit --layers 2 --batch_size 128 --log --epochs 10 --optimizer momentum --exp_id 10
echo exp 10 done
python run_exp_vit.py --model vit --layers 4 --batch_size 128 --log --epochs 10 --optimizer adamw --exp_id 11
echo exp 11 done
python run_exp_vit.py --model vit --layers 6 --batch_size 128 --log --epochs 10 --optimizer adamw --exp_id 12
echo exp 12 done
python run_exp_vit.py --model vit --layers 6 --batch_size 128 --log  --epochs 10 --optimizer adamw --block postnorm --exp_id 13
echo exp 13 done