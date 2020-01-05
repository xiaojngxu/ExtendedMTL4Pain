# ExtendedMTL4Pain

This is an implementation of 
Xu, X., Huang, J.S., & de Sa, V.R. (2019). Pain Evaluation in Video using Extended Multitask Learning from Multidimensional Measurements. Proceedings of Machine Learning Research, (Machine Learning for Health ML4H at NeurIPS 2019)

Install: python2.7, pytorch, cv2, matlab (for face detection)

Pre-processing:
1. Download UNBCMcMaster shoulder pain dataset in ./data/
2. Download vgg_face_matconvnet code in ./data/ from https://www.robots.ox.ac.uk/~vgg/software/vgg_face/ for face detection and cropping
3. Run process_shoulder_pain_data.m in ./data/vgg_face_matconvnet/ to process shoulder pain data and store cropped face images in ./data/UNBCMcMaster_cropMat/

Model training and test:
1. Download vgg_face_torch.tar.gz from http://www.robots.ox.ac.uk/~vgg/software/vgg_face/  put the model with weights VGG_FACE.t7 to current directory
2. Run detect_pscore.py in ./shoulder_pain_detection/PSPIAU/ to train models for stage 1
3. Run predict_VAS.py in ./shoulder_pain_detection/VAS3_using_PSPIinAUpred/ to train models for stage 2
4. plot_results.py includes stage 3 and prints results
