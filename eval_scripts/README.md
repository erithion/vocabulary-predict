### Additional data downloads
* Bokmål skip gram vectors http://vectors.nlpl.eu/repository/11/104.zip
* Norsk aviskorpus fra https://www.nb.no/sprakbanken og med den direkte lenken http://www.nb.no/sbfil/tekst/norsk_aviskorpus.zip 
### Evaluation results
* Model: Skip-gram word2vec + Binary classifier
* Skewed data: 15-25% positives of the overall words 
* Classifier: SVM (gaussian & linear krnls) and linear regression checked; the best score gives SVM gaussian kernel
    * C = 1
    * gamma = 0.01; 

* Words for tests were sampled uniformly
* Bigger values of gamma give better learning curve, but worse prediction of new words (generalisation) which was the key goal
    
    * Example

        Evaluation (test-set)                                     
                precision    recall  f1-score   support     
                                                          
            0       0.95      0.97      0.96      1106     
            1       0.90      0.83      0.86       335     
                                                          
    micro avg       0.94      0.94      0.94      1441     
                                                          
        Evaluation vs. training word number: 721 / 6481                     
        New eval words vs. total eval words: 153 / 721 (21.22%)             
        1-new eval words vs. 1-total eval words: 28 / 172 (16.28%)          
                                                                         
        Predicted vs. planned: 146 / 172 (84.88%)                           
        Predicted vs. planned (new words only): 2 / 28 (7.14%)              
            + 20 words by virtue of generalisation                           
            
    ![Learning curve](https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/learning_curve.png "Learning curve")
    ![PCA view](https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/pca.png "PCA view")
            