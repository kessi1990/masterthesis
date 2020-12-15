#!/usr/bin/env bash
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat_fc Seaquest 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Seaquest 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat_fc Seaquest 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Seaquest 104

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat_fc Pong 104
# sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Pong 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat_fc Pong 104
# sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Pong 104

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat_fc BeamRider 104
# sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat BeamRider 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat_fc BeamRider 104
# sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat BeamRider 104

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat_fc Breakout 104
# sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Breakout 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat_fc Breakout 104
# sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Breakout 104

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat_fc Krull 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Krull 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat_fc Krull 104
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Krull 104