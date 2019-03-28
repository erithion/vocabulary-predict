## Additional data downloads
* Bokmål skip gram vectors http://vectors.nlpl.eu/repository/11/104.zip
* Norsk aviskorpus fra https://www.nb.no/sprakbanken og med den direkte lenken http://www.nb.no/sbfil/tekst/norsk_aviskorpus.zip 
## Evaluation results
* Model: Skip-gram word2vec + Binary classifier
* Skewed data: 15-25% positives of the overall words 
* Classifier: SVM (gaussian & linear krnls) and linear regression checked; the best score gives SVM gaussian kernel
    * C = 1
    * gamma = 0.01; 

* Words sampled by uniform dstrb
* Bigger values of gamma give better learning curve, but worse prediction of neighboring words (generalisation) which was the key goal
    
### Example (gamma=1e-2)
```
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
   ``` 
<details>
 <summary>Learning curve. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/learning_curve.PNG" width="640" height="480">
</details>
<details>
 <summary>Learning curve. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/PCA.PNG" width="640" height="480">
</details>

### Example (gamma=1e9)
```
        Evaluation (test-set)                                                            
              precision    recall  f1-score   support                            
                                                                                 
            0       0.97      1.00      0.98       966                            
            1       1.00      0.93      0.96       465                            
                                                                                 
    micro avg       0.98      0.98      0.98      1431                            

        Evaluation vs. training word number: 716 / 6437                                  
        New eval words vs. total eval words: 27 / 716 (3.77%)                            
        1-new eval words vs. 1-total eval words: 6 / 224 (2.68%)                         
                                                                                 
        Predicted vs. planned: 218 / 224 (97.32%)                                        
        Predicted vs. planned (new words only): 0 / 6 (nan%)                             
            + 0 words by virtue of generalisation
```
<details>
 <summary>Learning curve. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/learning_curve_gamma_big.PNG" width="640" height="480">
</details>
<details>
 <summary>Learning curve. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/PCA_gamma_big.PNG" width="640" height="480">
</details>
