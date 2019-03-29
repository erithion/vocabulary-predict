## Additional data downloads
* Bokmål skip gram vectors http://vectors.nlpl.eu/repository/11/104.zip
* Norsk aviskorpus fra https://www.nb.no/sprakbanken og med den direkte lenken http://www.nb.no/sbfil/tekst/norsk_aviskorpus.zip 
## Evaluation results
* Model: Skip-gram word2vec + Binary classifier
* Skewed data: 15-25% positives of the overall words 
* Classifier: SVM (gaussian & linear krnls) and linear regression checked; the best score gives SVM gaussian kernel
* Words sampled by uniform dstrb
    
### Example (gamma=1e-2)
```
        Evaluation (test-set)                                              
              precision    recall  f1-score   support              
                                                                   
            0       0.96      0.97      0.97      1082              
            1       0.89      0.87      0.88       321              
                                                                   
    micro avg       0.95      0.95      0.95      1403              
                                                                   
        Evaluation vs. training word number: 702 / 6309                    
        New eval words vs. total eval words: 171 / 702 (24.36%)            
        1-new eval words vs. 1-total eval words: 26 / 169 (15.38%)         
                                                                   
        Predicted vs. planned: 148 / 169 (87.57%)                          
        Predicted vs. planned (new words only): 5 / 26 (19.23%)            
            + 14 words by virtue of generalisation                          
``` 
<details>
 <summary>Learning curve. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/learning_curve_gamma_small.PNG" width="640" height="480">
</details>
<details>
 <summary>Word embeddings approximation with PCA. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/PCA_gamma_small.PNG" width="640" height="480">
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
 <summary>Word embeddings approximation with PCA. Click to see the image</summary>

   <img src="https://github.com/erithion/resource/blob/master/vocabulary-predictor-pic/PCA_gamma_big.PNG" width="640" height="480">
</details>

## Summary
Bigger values of gamma produce better learning curve, but worse prediction of neighboring words (generalisation) which was the ultimate goal. Thus chosen hyper-parameters of SVM gaussian krnl:
* C = 1
* gamma = 1e-2
