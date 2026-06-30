#!/bin/bash

python3 experiment10.py ldpc_final_tesc/ \ 
    --gpu 0 \ 
    --throughput-mode \
    --decoder-api explicit \
    --grouping-mode explicit_codeblocks \
    --group-cbs 1 \
    --warmup-repeats 50 \
    --repeats 1000 \
    --timing-mode both \
    --csv final_tesc_aerial_1cb.csv
