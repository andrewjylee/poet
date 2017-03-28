# poet

train with:
python main.py --train_filename data/total.txt --num_epochs 30 --cell_type gru --batch_size 50 --num_steps 120 --dropout_prob_input 0.9 --dropout_prob_output 0.9 --state_size 512 --save_dir checkpoints/various_poets

TODO

change poem_length in write()
