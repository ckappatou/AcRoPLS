# AcRoPLS :classical_building:
AcRoPLS (**Ac**urate **Ro**bust **PLS**) provides a methodology to create accurate and robust PLS models based on solving a data-driven optimization problem that couples data pre-processing and model regression to a single optimization step. The accuracy objective is evaluated based on the performance of the generated model on predicting the model output on a test set. For the robustness objective, a novel metric based on the method of moments applied for different realizations of a known variability source evaluated again on a test set. <!-- For more information on the method, please refer to: [add paper link]().  --> 


## Requirements 

The optimization is performed using [ENTMOOT](https://github.com/cog-imperial/entmoot) and the PLS model is created using [pyphi](https://github.com/salvadorgarciamunoz/pyphi). Both these packages are required for the code to be excecuted. All code was run on python 3.8.5 with following package versions:

#### Packages 
|Package| Version|
|-------|--------|
|cycler | 0.10.0|
|matplotlib |       3.4.1|
|numpy |          1.20.2|
|pandas |         1.2.4|
|scikit-learn | 0.24.2|
|scipy |       1.6.3|
|openpyxl|3.0.10|

#### ENTMOOT Installation 
This requires version 0.2.2 of ENTMOOT. See list of requirements from https://github.com/cog-imperial, then install 0.2.2 of ENTMOOT using 

    pip install git+https://github.com/cog-imperial/entmoot.git@8e6e0fbfaba7823269065ee3c88d56d354167bb9

#### Running Examples

Navigate to the examples file in your terminal and run the desired example, e.g. 

    python3 InitialExample.py
    
<!--Note that the third example in [add paper link]() is not provided in this repository due to confidentiallity of the industrial dataset used for that particular case study.-->


## Contributors 

| Contributor      | Acknowledgements          |
| ---------------- | ------------------------- |
| Chryssa Kappatou     | This research is funded by an Engineering and Physical Sciences Research Council / Eli Lilly Prosperity Partnership (EPSRC EP/T005556/1) and by Eli Lilly \& Company|
