echo "{"
echo '"GRU crf ner":'
python evaluate.py model/gru_neural_crf_ner data/twitter_test.ner
echo ","
echo '"LSTM2D crf ner":'
python evaluate.py model/lstm2d_neural_crf_ner data/twitter_test.ner
echo ","
echo '"BASELINE crf ner":'
python evaluate.py model/_neural_crf_ner data/twitter_test.ner
echo "}"

