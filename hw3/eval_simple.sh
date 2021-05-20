echo "{"
echo '"GRU s pos":'
python evaluate.py model/gru_s_tagger_pos data/twitter_test.pos
echo ","
echo '"LSTM2D s pos":'
python evaluate.py model/lstm2d_s_tagger_pos data/twitter_test.pos
echo ","
echo '"BASELINE s pos":'
python evaluate.py model/_s_tagger_pos data/twitter_test.pos
echo ","


echo '"GRU s ner":'
python evaluate.py model/gru_s_tagger_ner data/twitter_test.ner
echo ","
echo '"LSTM2D s ner":'
python evaluate.py model/lstm2d_s_tagger_ner data/twitter_test.ner
echo ","
echo '"BASELINE s ner":'
python evaluate.py model/_s_tagger_ner data/twitter_test.ner
echo "}"
