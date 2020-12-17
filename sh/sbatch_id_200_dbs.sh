#!/usr/bin/env bash
#####################################
#              NO LSTM              #
#####################################
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 4 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 no-lstm 8 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 4 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 no-lstm 8 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 4 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 no-lstm 8 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 4 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 no-lstm 8 dot large True 200

#####################################
#                 ID                #
#####################################

sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 1 identity 1 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 2 identity 2 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 3 identity 3 dot large True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 concat_fc small True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 concat small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 general small True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 dot small True 200

sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 concat_fc large True 200
sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 concat large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 general large True 200
# sbatch --gres=gpu:0 no_lstm_dbs.sh 4 identity 4 dot large True 200