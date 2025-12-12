<a href='https://www.daytum.io/'><img src="https://github.com/daytum/logos/blob/master/daytum_logo_2019.png?raw=true" alt="Daytum" style="width: 250px;"/></a>

# 3-Day Subsurface Maching Learning, Spatial Data Analytics, and Geostatistics Course 

This repository contains files related to a training class dated 12/08/2025.

Instructor: Michael Pyrcz, The University of Texas at Austin

#### Course Summary

Building from fundamental probability and statistics, we cover entire spatial data analytics and geostatistics best practice workflows from data preparation through to decision making. We will accomplish this with,

* Interactive lectures / discussion to cover the basic concepts

* Demonstrations of methods and workflows in Python 

* Hands-on experiential learning with well-documented workflows for accessibility


#### Course Objectives

Spatial data analytics and geostatistics for building spatial prediction and uncertainty models.

You will learn:

* spatial data debiasing

* quantification and modeling of spatial continuity / correlation

* spatial estimation with uncertainty

* spatial simulation for subsurface resource forecasting

* checking spatial models

* decision making with spatial uncertainty models

#### Course Schedule

##### Spatial Data Analytics and Geostatistics 1-day Short Course

| Day | Time | Topic  | Objective | Notes | Demo | Interactive | e-book | Lecture |
|-|-|-|--|-|-|-|-|-|
| Day 1 | 8:00 AM - 8:30 AM     | Course Overview                        | Walk-through of the course plan, goals, methods and introductions            | [Overview](/pdfs/CourseOverview.pdf)         | | | | |
|       | 8:30 AM - 9:00 AM     | Introduction                           | Data analytics and geostatistics concepts                                    | [Introduction](/pdfs/Introduction.pdf)       | | | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_concepts.html) | [Lecture](https://www.youtube.com/watch?v=pxckixOlguA&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=1) |
|       | 9:00 AM - 10:00 AM    | Data Analytics                         | Multivariate statistical methods to support spatial modeling                 | [Notes](/Pyrcz_UTCourse/08_Bivariate_Correlation.pdf) | [Demo](/notebooks/multivariate_analysis.ipynb) | [Dashboard1](/notebooks/Interactive_Correlation_Coefficient.ipynb) [Dashboard2](/notebooks/Interactive_Correlation_Coefficient_Issues.ipynb) |  [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_bivariate.html)  | [Lecture](https://www.youtube.com/watch?v=wZwYEDqB4A4&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=21) |
|       | 10:00 AM - 11:00 AM   | Spatial Continuity Calculation         | Measuring spatial continuity with experimental variograms                    | [Notes](/Pyrcz_UTCourse/10_Spatial_Calc.pdf) | [Demo](/notebooks/variogram_calculation.ipynb) | [Dashboard](/notebooks/Interactive_Variogram_Calculation.ipynb) | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_variogram_calculation.html) | [Lecture](https://www.youtube.com/watch?v=j0I5SGFm00c&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=32) |
|       | 11:00 AM - 12:00 Noon | Spatial Continuity Modeling            | Variogram modeling for quantifying spatial continuity                        | [Notes](/Pyrcz_UTCourse/11_Spatial_Interpretation_Modeling.pdf) | [Demo](/notebooks/variogram_modeling.ipynb) | [Dashboard](/notebooks/Interactive_Variogram_Calculation_Modeling.ipynb) | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_variogram_calculation_and_modeling.html) | [Lecture](https://www.youtube.com/watch?v=Li-Xzlu7hvs&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=38) |
|       | 12:00 noon - 1:00 PM  | Lunch Break                            | | | | | |
|       | 1:00 PM - 2:00 PM     | Spatial Estimation                     | Introduce spatial estimators, theory and applications with kriging           | [Notes](/Pyrcz_UTCourse/12_Kriging.pdf)  | [Demo](/notebooks/kriging.ipynb) | [Dashboard](/notebooks/Interactive_Simple_Kriging.ipynb) | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_kriging.html) | [Lecture](https://www.youtube.com/watch?v=BCnivpSKF18&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=41) [Lecture2](https://www.youtube.com/watch?v=CVkmuwF8cJ8&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=42) |
|       | 2:00 PM - 2:30 PM     | Simulation and Uncertainty Modeling    | Stochastic realizations for uncertainty modeling                             | [Notes](/Pyrcz_UTCourse/13_Simulation.pdf) | [Demo](/notebooks/simulation.ipynb) | [Dashboard](/notebooks/Interactive_Simulation.ipynb) | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_simulation.html) | |
|       | 2:30 PM - 3:00 PM     | Advanced Simulation (Optional)         | Cosimulation for bivariate simulation models                                 | [Notes](/Pyrcz_UTCourse/16_Cosimulation.pdf) | | | | |
|       |                       |                                        | Indicator simulation                                                         | [Notes](/Pyrcz_UTCourse/14_Simulation_Indicator.pdf) | [Demo](/notebooks/sisim.ipynb) | | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_categorical_indicator_simulation.html) | [Lecture](https://www.youtube.com/watch?v=6mCfgbh7f2g&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=46) |
|       |                       |                                        | Multiple point and object-based simulation                                   | [Notes](/Pyrcz_UTCourse/15_Simulation_Facies.pdf) | | | | |
|       | 3:00 PM - 4:00 PM     | Model Checking                         | Essential quality assurance methods for spatial, geostatistical models       | [Notes](/Pyrcz_UTCourse/16b_Model_Checking.pdf) | [Demo](/notebooks/model_checking.ipynb) | | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_model_checking.html) | [Lecture](https://www.youtube.com/watch?v=AVms8JoUWXc&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=49) |
|       | 4:00 PM - 5:00 PM     | Decision Making with Uncertainty       | Making the best decision in the precense of uncertainty                      | [Notes](/Pyrcz_UTCourse/16c_Decision_Making.pdf) |  | [Dashboard](/notebooks/Interactive_Decision_Making.ipynb) | [Book](https://geostatsguy.github.io/GeostatsPyDemos_Book/GeostatsPy_decision_making.html) | [Lecture](https://www.youtube.com/watch?v=Yu9jDlqBrJE&list=PLG19vXLQHvSB-D4XKYieEku9GQMQyAzjJ&index=50) |

##### Machine Learning 2-day Short Course

| Day | Time | Topic  | Objective | Notes | Demo | Interactive | e-book |
|-|-|-|--|-|-|-|-|
| Day 2 | 8:00 AM - 8:30 AM     | Course Overview                        | Review schedule only                                                         | [Overview](/pdfs/CourseOverview.pdf)    | | | |
|       | 8:30 AM - 10:00 AM    | Probability                            | Frequentist and Bayesian probability approaches                              | [Notes](/Pyrcz_UTCourse/02_Probability.pdf) | | [Dashboard](/notebooks/Interactive_Sivia_Coin_Toss.ipynb) | |
|       | 11:00 AM - 12:00 PM   | Data Preparation                       | Data debiasing methods to correct for sampling bias                          | [Notes](/Pyrcz_UTCourse/04_DataPrep.pdf) | [Demo](/notebooks/declustering.ipynb)| [Dashboard](/notebooks/Interactive_Declustering.ipynb) | | |
|       |                       |                                        | Introduction to bootstrap for uncertainty modeling                           | [Notes](/Pyrcz_UTCourse/04_DataPrep.pdf) | [Demo](/notebooks/bootstrap.ipynb) | [Dashboard](/notebooks/Interactive_Bootstrap.ipynb) | |
|       | 12:00 noon - 1:00 PM  | Lunch Break                            | | | | | |
|       | 1:00 PM - 2:00 PM     | Feature Imputation                     | Dealing with missing data                                                    | [Notes](/Pyrcz_UTCourse/05d_Feature_Imputation.pdf) | | | |
|       | 2:00 PM - 3:00 PM     | Cluster Analysis                       | k-means clustering                                                           | [Notes](/Pyrcz_UTCourse/07_Clustering.pdf) | | | |
|       | 3:00 PM - 4:00 PM     | Advanced Cluster Analysis              | Density-based and spectral clustering                                        | [Notes](/Pyrcz_UTCourse/07b_Clustering_Advanced.pdf) | | | |
|       | 4:00 PM - 5:00 PM     | Dimensionality Reduction               | Principal components analysis                                                | [Notes](/Pyrcz_UTCourse/Dimensional_Reduction.pdf) | | | |
| Day 3 | 8:00 AM - 9:00 AM     | Predictive Machine Learning            | Concepts and workflows for predictive machine learning                       | [Notes](/Pyrcz_UTCourse/06_Machine_Learning.pdf) | | | |
|       | 9:00 AM - 10:00 AM    | Linear Regression                      | Start with simple linear prediction models                                   | [Notes](/Pyrcz_UTCourse/09_LinearRegression.pdf)
|       | 9:00 AM - 10:00 AM    | k-Nearest Neighbors                    | Lazy learning with a mapping analogy                                         | [Notes](/Pyrcz_UTCourse/11_kNearestNeighbours.pdf) | [Demo](notebooks/kNearestNeighbour.ipynb) | | |
|       | 10:00 AM - 11:00 AM   | Naive Bayes                            | Bayesian classification model                                                | [Notes](/Pyrcz_UTCourse/12_NaiveBayesClassifier.pdf) | [Demo](notebooks/NaiveBayes.ipynb) | | |
|       | 11:00 AM - 12:00 noon | Decision Tree                          | Simple model that extends to powerful ensemble methods                       | [Notes](/Pyrcz_UTCourse/14_DecisionTree.pdf) | [Demo](notebooks/NaiveBayes.ipynb) | | |
|       | 12:00 noon - 1:00 PM  | Lunch Break                            | | | | | |
|       | 1:00 PM - 2:00 PM     | Bagging and Random Forest              | Averaging over trees to reduce model variance                                | [Notes](/Pyrcz_UTCourse/15_EnsembleTree.pdf) | [Demo](notebooks/NaiveBayes.ipynb) | | |
|       | 2:00 PM - 3:00 PM     | Gradient Boosting                      | Additive weak learners to avoid overfit                                      | [Notes](/Pyrcz_UTCourse/15b_Gradient_Boosting.pdf) | [Demo](notebooks/NaiveBayes.ipynb) | | |
|       | 3:00 PM - 4:00 PM     | Neural Networks                        | Powerful deep learning methods                                               | [Notes](/Pyrcz_UTCourse/17_Neural_Networks.pdf) | [Demo](notebooks/NaiveBayes.ipynb) | | |
|       | 4:00 PM - 4:30 PM     | Conclusions and Wrap-up                | Summarize and discuss                                                        | [Notes](/Pyrcz_UTCourse/17_Neural_Networks.pdf) | [Demo](notebooks/NaiveBayes.ipynb) | | |

This is a nominal schedule. Note, we are learning and not schedule-driven; therefore the course delivery will adjust for the needs of the class. 

#### Beyond the Course

**There is Much More** – the building blocks can be reimplemented and expanded to address various other problems, opportunities. There is much more that we could cover,

* Additional Theory

* More Hands-on / Experiential

* Workflow Development

* Basics of Python / R

* Advanced Data Preparation

* Advanced Model QC

* Methods to Integrate More Geoscience and Engineering

* Integration of Machine Learning Spatial Modeling

We are happy to discuss other, advanced courses and custom courses to meet your teams' educational needs to add value at work with data science.

Daytum's courses have been taken by employees at:<br><br>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Chevron_Logo.svg/918px-Chevron_Logo.svg.png" style="width: 40px;"/>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/d/da/Chord_Energy_logo.svg/1920px-Chord_Energy_logo.svg.png" style="width: 160px;"/>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/a/a7/Saudi_Aramco_logo.svg/1920px-Saudi_Aramco_logo.svg.png" style="width: 100px;"/>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/Equinor.svg/1280px-Equinor.svg.png" style="width: 90px;"/>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://1000logos.net/wp-content/uploads/2020/09/ConocoPhillips-Logo.png" style="width: 100px;"/>
&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://1000logos.net/wp-content/uploads/2016/10/BP-Logo.png" style="width: 100px;"/>
<p><i>&copy; Copyright daytum 2025. All Rights Reserved</i></p>
