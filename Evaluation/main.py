import sys
from .runner import main

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: python -m Evaluation.main <files_folder> <output_folder> <start_index>')
    else:
        main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
