# declare -a bit_lst=("16" "14" "12" "10" "8" "7" "6" "5" "4" "3" "2")
declare -a bit_lst=("3" "4" "5" "6" "7" "8" "12" "16")
# declare -a bit_lst=("7" "6")

for bit_wid in "${bit_lst[@]}"; do
    python -W ignore roberta_base_analyzer.py -e -aq ${bit_wid}.0
# python -W ignore roberta_sst2_analyzer.py -e -aq ${bit_wid}.0
    tar -cvf params_all_${bit_wid}.tar.gz params/*_all.npy
    rm params/*_all.npy
done
