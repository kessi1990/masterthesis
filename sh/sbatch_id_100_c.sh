#!/usr/bin/env bash
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Seaquest 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead general Seaquest 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead dot Seaquest 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Seaquest 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn general Seaquest 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn dot Seaquest 100

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Pong 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead general Pong 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead dot Pong 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Pong 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn general Pong 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn dot Pong 100

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat BeamRider 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead general BeamRider 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead dot BeamRider 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat BeamRider 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn general BeamRider 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn dot BeamRider 100

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat Breakout 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead general Breakout 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead dot Breakout 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat Breakout 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn general Breakout 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn dot Breakout 100

sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead concat SpaceInvaders 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead general SpaceInvaders 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False cead dot SpaceInvaders 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn concat SpaceInvaders 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn general SpaceInvaders 100
sbatch --partition=Gobi,Luna,All --gres=gpu:0 --exclusive batch_new.sh False darqn dot SpaceInvaders 100
