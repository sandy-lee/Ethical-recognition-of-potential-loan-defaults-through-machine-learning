# Mod 5 Project - Ethical recognition of potential loan defaults through machine learning

# Problem framing

Every year default loans cost the banking industry on average $10,000 USD according to the Bank of England Statistics pages and with 1 in 5 calls to the Samaritans in the UK now dues to "extreme financial distress", it is imperative we take advantage of advances in data processing and machine leanring to ensure we are only lending money to people who can afford to repay.

We are fortunate in he banking industry to be able to have access to lot of data surrounding how loans are are approved and the results of those loans (either full repayment or default). We are also fortunate to be able to have access to different machine learning models to be able to look for patters in the data to provide insights into the types of questions we should be asking people who will potentially borrow money from us. As well as devise machine learning models that are able to predict with a higher degree of accuracy than our current methods in order to predict if a loan will be repaid or default.

# Success Metrics

According to our current loans dataset we get this right 70% of the time, therefore success is:

>1) Building a model that predicts defualt loans from our current data with an error less than 30%
>
>2) We are able to examine the current dataset, in order to find what factors most accurately predict when a loan will default, so we can improve our application process overall


# Winning Model

We have evaluated 3 models: logistic regression, a decision tree and a random forest. We optimised the hyperparamters for each and plotted a ROC curve in order to determine the performance. What we found was that the AUC metric increased with each model until we achived an AUC of 0.73 with the decision tree on the validation data set when doing 10 x K-fold. This is the best performing model out of all of the candidates we test (which was around 3,000 different models) and next steps are to determine what is the threshhold for our model performance taking into account business imperatives.

# Next Steps

Next steps are to review our threshold calculation above as it doesn't seem correct and to review the performance of the model against the test dataset, which is something we would have done if we had more time. Assuming satisfactory performance we would then put this into production.
