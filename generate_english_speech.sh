
# Train data
for i in 00 01 02;do
python generate_english_speech_data.py \
        /your_path/data-english/split-files/clg8_tsv_${i}_st2852.tsv \
        /your_path/data-english/split-files/json-file/clg8_tsv_$i.json \
        /your_path/data-english/split-files/json-file/clg8_tsv_$i.err.txt     
done  


# BAE19 dev data
python generate_english_speech_data.py \
        /your_path/data-english/ABCN_bea19_dev.tsv \
        /your_path/data-english/json-file/ABCN_bea19_dev.json \
        /your_path/data-english/json-file/ABCN_bea19_dev.err.txt
        
        

# CoLL14 test data
python generate_english_speech_data.py \
        /your_path/data-english/coll14_test.tsv \
        /your_path/data-english/json-file/coll14_test.json \
        /your_path/data-english/json-file/coll14_test.err.txt
        
        

# BAE19 test data
python generate_english_speech_data.py \
        /your_path/data-english/ABCN_bea19_test.tsv \
        /your_path/data-english/json-file/ABCN_bea19_test.json \
        /your_path/data-english/json-file/ABCN_bea19_test.err.txt
