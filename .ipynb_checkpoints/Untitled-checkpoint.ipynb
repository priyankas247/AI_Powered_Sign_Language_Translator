{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "548a29ed-54f4-4ffe-92b4-2854d539ab30",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: opencv-python in c:\\users\\priya\\anaconda3\\lib\\site-packages (4.11.0.86)Note: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "Requirement already satisfied: mediapipe in c:\\users\\priya\\anaconda3\\lib\\site-packages (0.10.21)\n",
      "Requirement already satisfied: numpy in c:\\users\\priya\\anaconda3\\lib\\site-packages (1.26.4)\n",
      "Requirement already satisfied: absl-py in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (2.2.0)\n",
      "Requirement already satisfied: attrs>=19.1.0 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (23.1.0)\n",
      "Requirement already satisfied: flatbuffers>=2.0 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (25.2.10)\n",
      "Requirement already satisfied: jax in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (0.5.3)\n",
      "Requirement already satisfied: jaxlib in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (0.5.3)\n",
      "Requirement already satisfied: matplotlib in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (3.8.4)\n",
      "Requirement already satisfied: opencv-contrib-python in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (4.11.0.86)\n",
      "Requirement already satisfied: protobuf<5,>=4.25.3 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (4.25.6)\n",
      "Requirement already satisfied: sounddevice>=0.4.4 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (0.5.1)\n",
      "Requirement already satisfied: sentencepiece in c:\\users\\priya\\anaconda3\\lib\\site-packages (from mediapipe) (0.2.0)\n",
      "Requirement already satisfied: CFFI>=1.0 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from sounddevice>=0.4.4->mediapipe) (1.16.0)\n",
      "Requirement already satisfied: ml_dtypes>=0.4.0 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from jax->mediapipe) (0.5.1)\n",
      "Requirement already satisfied: opt_einsum in c:\\users\\priya\\anaconda3\\lib\\site-packages (from jax->mediapipe) (3.4.0)\n",
      "Requirement already satisfied: scipy>=1.11.1 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from jax->mediapipe) (1.13.1)\n",
      "Requirement already satisfied: contourpy>=1.0.1 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (1.2.0)\n",
      "Requirement already satisfied: cycler>=0.10 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (0.11.0)\n",
      "Requirement already satisfied: fonttools>=4.22.0 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (4.51.0)\n",
      "Requirement already satisfied: kiwisolver>=1.3.1 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (1.4.4)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (23.2)\n",
      "Requirement already satisfied: pillow>=8 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (10.3.0)\n",
      "Requirement already satisfied: pyparsing>=2.3.1 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (3.0.9)\n",
      "Requirement already satisfied: python-dateutil>=2.7 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from matplotlib->mediapipe) (2.9.0.post0)\n",
      "Requirement already satisfied: pycparser in c:\\users\\priya\\anaconda3\\lib\\site-packages (from CFFI>=1.0->sounddevice>=0.4.4->mediapipe) (2.21)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\priya\\anaconda3\\lib\\site-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.16.0)\n"
     ]
    }
   ],
   "source": [
    "pip install opencv-python mediapipe numpy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "95d7441f-4815-487f-9f18-ecc833df4a40",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MediaPipe is working!\n"
     ]
    }
   ],
   "source": [
    "import mediapipe as mp\n",
    "print(\"MediaPipe is working!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "feb7180d-e0f6-482e-866d-2355f61f85cc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import mediapipe as mp\n",
    "from IPython.display import display, clear_output\n",
    "import time\n",
    "\n",
    "# Initialize MediaPipe Hand module\n",
    "mp_hands = mp.solutions.hands\n",
    "mp_draw = mp.solutions.drawing_utils\n",
    "hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)\n",
    "\n",
    "# Start video capture\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "while cap.isOpened():\n",
    "    ret, frame = cap.read()\n",
    "    if not ret:\n",
    "        break\n",
    "\n",
    "    # Convert BGR to RGB (required by MediaPipe)\n",
    "    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\n",
    "    results = hands.process(rgb_frame)\n",
    "\n",
    "    # Draw hand landmarks\n",
    "    if results.multi_hand_landmarks:\n",
    "        for hand_landmarks in results.multi_hand_landmarks:\n",
    "            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)\n",
    "\n",
    "    # Display the output\n",
    "    cv2.imshow(\"Hand Tracking\", frame)\n",
    "\n",
    "    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit\n",
    "        break\n",
    "\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c02a8d87-517b-4052-b83c-7df4776b4129",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
