CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ks50'  --tta-method 'None' --pretrain_path "/path/to/cav_mae_ks50.pth" --corruption-modality 'both' --audio_c_type crowd

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ks50'  --tta-method 'None' --pretrain_path "/path/to/cav_mae_ks50.pth" --corruption-modality 'both' --audio_c_type gaussian_noise

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ks50'  --tta-method 'None' --pretrain_path "/path/to/cav_mae_ks50.pth" --corruption-modality 'both' --audio_c_type rain

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ks50'  --tta-method 'None' --pretrain_path "/path/to/cav_mae_ks50.pth" --corruption-modality 'both' --audio_c_type thunder

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ks50'  --tta-method 'None' --pretrain_path "/path/to/cav_mae_ks50.pth"  --corruption-modality 'both' --audio_c_type traffic

CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'ks50'  --tta-method 'None' --pretrain_path "/path/to/cav_mae_ks50.pth"  --corruption-modality 'both' --audio_c_type wind



