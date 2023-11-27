python -u main.py --model_name aitm --tgt_dataset_name AliExpress_NL
python -u main.py --model_name ple --weight_decay 1e-2 --tgt_dataset_name AliExpress_NL
python -u main.py --model_name omoe --weight_decay 1e-3 --learning_rateda 0.1 --tgt_dataset_name AliExpress_NL
python -u main.py --model_name aitm --tgt_dataset_name AliExpress_ES
python -u main.py --model_name ple --weight_decay 1e-2 --tgt_dataset_name AliExpress_ES
python -u main.py --model_name omoe --weight_decay 1e-3 --learning_rateda 0.1 --tgt_dataset_name AliExpress_ES
python -u main.py --model_name aitm --tgt_dataset_name AliExpress_FR
python -u main.py --model_name ple --weight_decay 1e-2 --tgt_dataset_name AliExpress_FR
python -u main.py --model_name omoe --weight_decay 1e-3 --learning_rateda 0.1 --tgt_dataset_name AliExpress_FR
python -u main.py --model_name aitm --tgt_dataset_name AliExpress_US
python -u main.py --model_name ple --weight_decay 1e-2 --tgt_dataset_name AliExpress_US
python -u main.py --model_name omoe --weight_decay 1e-3 --learning_rateda 0.1 --tgt_dataset_name AliExpress_US
python -u main.py --model_name mmoe --tgt_dataset_name AliExpress_NL
python -u main.py --model_name sharedbottom --lr_e 0.5 --tgt_dataset_name AliExpress_NL
python -u main.py --model_name mmoe --tgt_dataset_name AliExpress_ES
python -u main.py --model_name sharedbottom --lr_e 0.5 --tgt_dataset_name AliExpress_ES
python -u main.py --model_name mmoe --tgt_dataset_name AliExpress_FR
python -u main.py --model_name sharedbottom --lr_e 0.5 --tgt_dataset_name AliExpress_FR
python -u main.py --model_name mmoe --tgt_dataset_name AliExpress_US
python -u main.py --model_name sharedbottom --lr_e 0.5 --tgt_dataset_name AliExpress_US

