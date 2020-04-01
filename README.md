# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>
(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2020



+ Team Group 1

+ Team Members: 
    +  Zhang, Zhiyuan zz2677@columbia.edu
    + Feng, Kangli kf2616@columbia.edu
    + Blum, Jacquelyn jeb2266@columbia.edu
    + Li, Jia jl5520@columbia.edu
    + Lin, Hongshan hl3353@columbia.edu
 
 
+ Project summary:  

In this project, we created a classification engine for facial emotion recognition. We were provided with a set of 2,500 facial images with 22 different emontions.The project purpose is to improve the accuracy and reduce the computation time of our model.For the feature extraction, we compuated pairwise distances between total unique 78 fiducial points with 3003 features. For baseline model training: we selected GBM with default parmater setting and achieved 43.8% accuracy, then we tuned our model by using different parametrs and cross validation to get our improved baseline model with 48.6% accuracy. For advanced model, We tried other models like XGBOOST, Random Forest, Residual Networks 50, Densely-connected Neural Network, Logistic Regression and Decision Tree to compare with our baseline model GBM. Finally we decided to use Densely-connected Neural Network as our final model with 52.6% accuracy. 

	
**Contribution statement**: 
+ Zhang, Zhiyuan: Designed the pipline for reading data, feature extraction and exectued the model. Constructed different models incuding GBM, XGBoost, KNN, Random Forest, Logistic Regression, SVM(linear, rgb, poly kernal) ,tuning the hyperparameter, cross validation and integrated the choosen model into main file. Prepared for the presentation(PPT and recording Video) Completed the project summary.
+ Kangli Feng: Conducted images cropping  to remove redundant image information such as background, clothes and hair. Converted processed data into h5 file for Residual Network model input. Built ResNet50 model with identity blocks and convolutional blocks. Extracted features from 78 fiducial points of each image by calculating pairwise distances for densely-connected Neural Networks input. Built three different densely-connected Neural Networks and achieved average test accuracy of 53%. Combined baseline model and advanced model into `Main.ipynb`.
+ Li Jia: Tuning hyperparameter for XGBoost
+ Blum, Jacquelyn:Participated in meeting and project, github arrangement.
+ Hongshan Lin: Tried different classfication model such as XGBoost, Random Forest, and GBM. Finally choose GBM as final model.Tuned the baseline model with different parameter to improve the accuracy from 43.8% to 45.5%. Edit Readme and final main notebook.


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
