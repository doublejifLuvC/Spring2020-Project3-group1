# Project: Can you recognize the emotion from an image of a face? 
<img src="figs/CE.jpg" alt="Compound Emotions" width="500"/>
(Image source: https://www.pnas.org/content/111/15/E1454)

### [Full Project Description](doc/project3_desc.md)

Term: Spring 2020



+ Team ##
+ Team members
      1. Zhang, Zhiyuan zz2677@columbia.edu
      2. Blum, Jacquelyn jeb2266@columbia.edu
      3. Feng, Kangli kf2616@columbia.edu
      4. Li, Jia jl5520@columbia.edu
      5. Lin, Hongshan hl3353@columbia.edu
 
 
+ Project summary:In this project, we created a classification engine for facial emotion recognition. 
We were provided with a set of 2,500 facial images with 22 different emontions.The project purpose is to improve the accuracy and reduce the computation time of our model.For the features extraction, we compuated pairwise distances between total uniq 78 fiducial points with 3003 features. For model training: we first select GBM as our baseline model with default parmater setting and 44% accuracy, then we tuned our model by using different parametrs and cross validation to get our improved baseline model with 48% accuracy. We also tried other models like XGBOOST, Random Forest, Neural Network, Logistic Regression and Decision Tree to compare with our baseline model GBM, finally we decided to use [xxxxxxx] as our final model with [xxxxxx] accuracy. 

	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) All team members contributed equally in all stages of this project. All team members approve our work presented in this GitHub repository including this contributions statement.

Zhang, Zhiyuan:

Li Jia:

Blum, Jacquelyn:

Feng, Kangli：

Hongshan Lin: Tried different classfication model such as XGBoost, Random Forest, and GBM. Finally choose GBM as final model.Tuned the baseline model with different parameter to improve the accuracy from 44% to 48%. Edit Readme and final main notebook.


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
