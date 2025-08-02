source qa-env/bin/activate

start_index=${1:-0}


python local_filtering.py  ../output/final/qa-eval ../output/final/local_95_2 "$start_index"