screen -d -m python stdmae/run.py --cfg='stdmae/TMAE_PEMS03.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/TMAE_PEMS04.py' --gpus='0'
screen -d -m python stdmae/run.py --cfg='stdmae/TMAE_PEMS07.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/TMAE_PEMS08.py' --gpus='0'
screen -d -m python stdmae/run.py --cfg='stdmae/SMAE_PEMS03.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/SMAE_PEMS04.py' --gpus='0'
screen -d -m python stdmae/run.py --cfg='stdmae/SMAE_PEMS07.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/SMAE_PEMS08.py' --gpus='0'

cp checkpoints/TMAE_200/064b0e96c042028c0ec44856f9511e4c/TMAE_best_val_MAE.pt mask_save/TMAE_PEMS04_864.pt

screen -d -m python stdmae/run.py --cfg='stdmae/STDMAE_PEMS04.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/STDMAE_PEMS03.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/STDMAE_PEMS08.py' --gpus='0'
screen -d -m python stdmae/run.py --cfg='stdmae/STDMAE_PEMS07.py' --gpus='0' 
-------------------------------------------------------------------------------------
screen -d -m python stdmae/run.py --cfg='stdmae/STMAE_PEMS08.py' --gpus='0' 

screen -d -m python stdmae/run.py --cfg='stdmae/SMAE_SD.py' --gpus='4,5' 
screen -d -m python stdmae/run.py --cfg='stdmae/TMAE_SD.py' --gpus='0' 
screen -d -m python stdmae/run.py --cfg='stdmae/PEMS03_PEMS04.py' --gpus='3' 

cp checkpoints/SMAE_200/SD/SMAE_best_val_MAE.pt mask_save/SMAE_SD_864.pt
cp checkpoints/TMAE_200/SD/TMAE_best_val_MAE.pt mask_save/TMAE_SD_864.pt
screen -d -m python stdmae/run.py --cfg='stdmae/STDMAE_SD.py' --gpus='5' 
screen -d -m python stdmae/run.py --cfg='stdmae/SD_PEMS04.py' --gpus='6' 

screen -d -m python stdmae/run.py --cfg='stdmae/GWNet_PEMS04.py' --gpus='1,2,3,4,5' 

-------------------------------------------------------------------------------------
# Trans to PEMS04 with simple MLP
# Trans_MLP_Simple Trans_PEMS03_Simple Trans_PEMS04_Simple Trans_PEMS08_Simple Trans_SD_Simple
screen -d -m python stdmae/run.py --cfg='stdmae/Trans_SD_Simple.py' --gpus='7' 
screen -d -m python stdmae/run.py --cfg='stdmae/Trans_PEMS08_Simple.py' --gpus='6' 
screen -d -m python stdmae/run.py --cfg='stdmae/Trans_PEMS04_Simple.py' --gpus='5' 
screen -d -m python stdmae/run.py --cfg='stdmae/Trans_PEMS03_Simple.py' --gpus='4' 
screen -d -m python stdmae/run.py --cfg='stdmae/Trans_MLP_Simple.py' --gpus='3' 