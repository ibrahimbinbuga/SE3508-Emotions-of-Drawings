# SE3508-Emotions-of-Drawings

## Project Info

*Course*: SE3508 – Introduction to Artificial Intelligence  
*Instructor*: Dr. Selim Yılmaz  
*Date*: 20.06.2025

---

## Contributors

- İbrahim Binbuğa – 210717007  
- Fethiye Sarı – 210717036

---

This project is a final project for the SE3508 Introduction to Artificial Intelligence course.

This project uses a Convolutional Neural Network (CNN) model (VGG16) to analyze children's drawings and classify them into four emotional categories: *Happy, **Sad, **Angry, and **Scared*.

---

This repository also contains the detailed PDF report in the root directory.

---

## Project Structure

- *Frontend*: React-based web app for uploading drawings and viewing results.
- *Backend*: Python Flask API running app.py for inference.
- *Model*: Pre-trained VGG16 used for transfer learning + handcrafted visual features.
- *Dataset*: Augmented children's drawings from Kaggle.

---

## Technologies Used

- Python, Flask
- React, JavaScript
- TensorFlow, Keras, OpenCV
- VGG16, ImageDataGenerator

---

## Emotion Categories

- Happy
- Sad
- Angry
- Scared

---

## How to Run the Project

### Backend (Flask API)

1. Navigate to the backend directory:
    ```
    bash
    cd backend 
    ```
   

2. Create a virtual environment and activate it:
    ```
    bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```
    bash
    pip install -r requirements.txt
    ```
   

4. Run the backend server:
    ```
    bash
    python app.py
    ```

---

### Frontend (React App)

1. Navigate to the frontend directory:
    ```
    bash
    cd frontend
    ```

2. Install dependencies:
    ```
    bash
    npm install
    ```

3. Start the development server:
    ```
    bash
    npm start
    ```   

---

## Model Performance

- Overall Accuracy: *57.25%*
- Best Performance: *Happy* class (~66%)
- Model: VGG16 + handcrafted features (color avg, edge density, number of faces)
- Data Augmentation: Rotation, shear, zoom, flips, normalization

---

## Application Preview

The user uploads a child’s drawing → the model analyzes it → returns the predicted emotion in real time with a clean UI.

---

## Sources

- Dataset: [Kaggle – Children Drawings](https://www.kaggle.com/datasets/vishmiperera/children-drawings/data)
- VGG16 Docs: [GeeksforGeeks](https://www.geeksforgeeks.org/vgg-16-cnn-model/)
- Psychology & Emotion Analysis Articles:
  - https://www.altugpsikoloji.com/post/cocuk-resimlerinin-gizemli-dunyas%C4%B1-resim-analizi
  - https://dergipark.org.tr/tr/download/article-file/3212762

---

## Credit
*This project was completed as part of the SE3508 Introduction to Artificial Intelligence course, instructed by Dr. Selim Yılmaz, Department of Software Engineering at Muğla Sıtkı Koçman University, 2025.*

*Note: This repository must not be used by students in the same faculty in future years—whether partially or fully—as their own submission. Any form of code reuse without proper modification and original contribution will be considered by the instructor a violation of academic integrity policies.*