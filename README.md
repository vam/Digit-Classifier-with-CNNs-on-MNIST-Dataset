MNIST Image classification model for AI-ML Lab Assignment 1

Note:- I used ubuntu for execution.


#Files

1)CNN_CODE.ipynb:In this I used CNN model to train MNIST dataset to make pre-trained model(digit_classifier_model.h5).

2)digit_classifier_model.h5:First it will be Pretrained model later after retraining pretrained model and incorrect predictions it will become fine tuned model.

3)streamlit_app.py :Developed a Streamlit application that allows users to upload images and see the model's predictions.Incorporate the trained CNN model into the Streamlit app for real-time predictions.

4)fine_tune.py :Code containing functions for scheduled fine-tuning of the pre-trained model based on user feedback.
 
5)incorrect_predictions:Folder to store images that resulted in incorrect predictions.

6)train.csv:it contain MNIST training dataset.


#Procedure
Task-1
Install required libraries and modules as mentioned in requirements.txt 
1)Run CNN_CODE.ipynb to know test accuracy and to get pretrained model(digit_classifier_model).


Task-2
1)Execution of streamlit_app.py :- In this code at 10th line, mention the path at which you have saved the pre trained model.Now execute the code in the terminal using the command:

   streamlit run streamlit_app.py
   
2)After execution web application will open there it will ask you to browse files
we need choose image file.


Task-3
1)it will predict number present on image and it will ask correct or incorrect.
    i)if actual label is equal to predict label then we need select correct option and  we need to chooose another image.
    ii)if actual label is not equal to predict label.we need  to select incorrect then it will ask what is actual label.we need to mention actual 		label
2)Then incorrect predictions will store in incorrect_predictions file.


Task-4
1) Execution of  fine_tune.py :incorrect_predictions file should be present in same directory where fine_tune.py file present.In this code at 91th line, mention the path at which you have saved the train.csv and  at 98th line, mention the path at which you have saved the pre trained model(digit_classifier_model).Now execute the code in the terminal using the command:

ubuntu terminal :python3 fine_tune.py

2)you will new accuracy after fine tuning.
3)After this rerun streamlit run streamlitapp.py then browse same image which given wrong prediction.
4)Now it will give correct prediction due to fine tuning.




