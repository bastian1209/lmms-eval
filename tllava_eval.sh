PYTHONPATH=../TinyLLaVA_Factory python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model tinyllava \
    --model_args pretrained="/home/ryong/codes/TinyLLaVA_Factory/checkpoints/tiny-llava-phi-2-siglip-so400m-patch14-384-base-finetune_naclip_w_naive_concat_ol13", conv_mode=phi \
    --tasks mme,pope,scienceqa_img,ai2d,ocrbench,mmmu_val,mathvista_testmini,mmvet,textvqa_val \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix tllava_ovs \
    --output_path "./logs/" 2>&1 | tee results.txt