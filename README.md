# ECE50024_mini_challenge

File Descriptions:

1. faces2.py:
   This file uses MTCNN from facenet_pytorch Library to Identify faces in an inputted image. The train_small folder is fed into this file for the images, and it creates images with cropped faces and saves in a defined output folder. For this model the output folder for train_small is temp2, and for test folder is temp4.
    Note: Faces.py was the first file I made which had the same purpose but used the CV2 library with Haarcascade classifier to detect faces but the MTCNN model gave better results so this is chosen for the final submission.
2. train.py:
    This is the main file to train the model. The input folder used by this file is the temp2 folder which has all the cropped faces for the images in train_small folder. The model used here is the facenet model from the facenet_pytorch library. I added two layers at the end to convert the model to classify 100 different label. 
3. test.py: 
   this file is used to test the trained model on the test images in the test folder. But as the model requires the face cropped images we first send the images in test folder to the faces2.py and store all the cropped images in temp4 folder. the predictions form the model are saved in a csv file called predictions_with_categories.csv. 
4. generate_submission.py:
   This file generates the submission.csv from the prediction_with_categories.csv in the format required for the submission on Kaggle. 
5. category.csv:
   contains the mappings of prediction labels to category labels.
6. train.csv:
   contains the mapping of truth labels to images in the train folder.
7. train_small.csv:
   contains the mapping of truth labels to images in the train_small folder.
8. sample_submission.csv:
   sample submission file.
9. train folder:
   contains the train images 
10. train_small folder:
    subset of the train folder
11. test folder:
    contains the test images
12. temp2 folder:
    contains the face cropped images from the train_small folder.
13. temp4 folder:
    contains the face cropped images from the test folder. 
14. facenet_epoch_10.pth:
    contains the model weights for the facenet model used after training for 10 epochs. 
15. submission.csv:
    submission file on Kaggle.
16. predictions_with_categories.csv:
    output file from test.py.

The submission zip file will not contain the image folder as they require too much space and will make the file size too large for submission. 
