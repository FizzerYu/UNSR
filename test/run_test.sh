###############################################################################################################
#pan_enhance_v6_ablation_concat_conv_miniatt_continue
CUDA_VISIBLE_DEVICES=0 python main.py --scale 4 --model wsr --pre_train ../experiment/wsr/model/model_best.pt --test_only --save_results --chop --save ../experiment/wsr/ --dir_data your_test_data_file/ --ext img --data_test B100

CUDA_VISIBLE_DEVICES=0 python main.py --scale 4 --model wsr --pre_train ../experiment/wsr/model/model_best.pt --test_only --save_results --chop --save ../experiment/wsr/ --dir_data your_test_data_file/ --ext img --data_test Set5

CUDA_VISIBLE_DEVICES=0 python main.py --scale 4 --model wsr --pre_train ../experiment/wsr/model/model_best.pt --test_only --save_results --chop --save ../experiment/wsr/ --dir_data your_test_data_file/ --ext img --data_test Set14

CUDA_VISIBLE_DEVICES=0 python main.py --scale 4 --model wsr --pre_train ../experiment/wsr/model/model_best.pt --test_only --save_results --chop --save ../experiment/wsr/ --dir_data your_test_data_file/ --ext img --data_test Urban100
