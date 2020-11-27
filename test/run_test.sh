###############################################################################################################
#pan_enhance_v6_ablation_concat_conv_miniatt_continue
CUDA_VISIBLE_DEVICES=3 mypython main.py --scale 4 --model wsr --pre_train /home/chenjunming/fizzer/PFFN/experiment/wsr/model/model_best.pt --test_only --save_results --chop --save /home/chenjunming/fizzer/PFFN/experiment/wsr/ --dir_data /titan_data2/lichangyu/sr_dataset/ --ext img --data_test B100

CUDA_VISIBLE_DEVICES=3 mypython main.py --scale 4 --model wsr --pre_train /home/chenjunming/fizzer/PFFN/experiment/wsr/model/model_best.pt --test_only --save_results --chop --save /home/chenjunming/fizzer/PFFN/experiment/wsr/ --dir_data /titan_data2/lichangyu/sr_dataset/ --ext img --data_test Set5

CUDA_VISIBLE_DEVICES=3 mypython main.py --scale 4 --model wsr --pre_train /home/chenjunming/fizzer/PFFN/experiment/wsr/model/model_best.pt --test_only --save_results --chop --save /home/chenjunming/fizzer/PFFN/experiment/wsr/ --dir_data /titan_data2/lichangyu/sr_dataset/ --ext img --data_test Set14

CUDA_VISIBLE_DEVICES=3 mypython main.py --scale 4 --model wsr --pre_train /home/chenjunming/fizzer/PFFN/experiment/wsr/model/model_best.pt --test_only --save_results --chop --save /home/chenjunming/fizzer/PFFN/experiment/wsr/ --dir_data /titan_data2/lichangyu/sr_dataset/ --ext img --data_test Urban100
