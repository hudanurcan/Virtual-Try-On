## Virtual Clothing Try-On Application
This project is a virtual clothing try-on application. Users can select clothing images from their computers and virtually try them on using a webcam.

## Technologies
The project is developed using the following technologies:

Tkinter: Used to create the user interface. It includes buttons, labels, and image display areas.

OpenCV: Used to capture and process webcam images. Additionally, an alpha channel is added to the clothing images to provide transparency.

MediaPipe: Used to detect body poses. The user's shoulder, hip, and other points are detected from the webcam feed, and the clothing is placed correctly on the body.

## How the Application Works?
After the user selects a clothing image, pose detection is performed through the webcam, and the clothing is placed correctly according to the user's body contours. Finally, the processed image is displayed on the screen using Tkinter.

This application allows users to virtually try on clothing, digitizing the shopping experience. Based on the user's body joints, clothing is placed on the upper part of the body.

## NOTES !!
The selected clothing image should be in PNG format with a transparent background. An example of a clothing image is shown below.

![ceket__1_-removebg-preview](https://github.com/user-attachments/assets/a57ceefe-8028-4ed9-8075-bb14529bb6a7)


