echo "{"
echo '"GRU crf pos":'
python evaluate.py model/gru_neural_crf_pos data/twitter_test.pos
echo ","
echo '"LSTM2D crf pos":'
python evaluate.py model/lstm2d_neural_crf_pos data/twitter_test.pos
echo ","
echo '"BASELINE crf pos":'
python evaluate.py model/_neural_crf_pos data/twitter_test.pos
echo "}"

