import React, { useState } from 'react';
import { Button, Container, Card, CardBody, CardTitle } from 'reactstrap';
import axios from 'axios';

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
    }
  };

  const emotionMessages = {
  angry: {
    title: "These may be the reasons:",
    reasons: [
      "Using dominant and dark colors such as red and black",
      "Presence of thick, hard and subdued lines",
      "Frowning, angry facial expressions (eyebrows down, mouth straight or downturned)",
      "Disorganized placement of figures or a complex layout on the page"
    ]
  },
  fear: {
    title: "These may be the reasons:",
    reasons: [
      "Preferring dark colors such as gray, black, dark blue",
      "Flickering, fuzzy or incomplete lines",
      "Surprised or frightened facial expressions (wide eyes, open mouth)",
      "Drawing figures small and placing them tucked into the corner of the page"
    ]
  },
  happy: {
    title: "These may be the reasons:",
    reasons: [
      "Preferring bright and vibrant colors such as yellow, light blue, pink",
      "Using clean, clear and stable lines",
      "Positive symbols such as smiling faces, sun, flower, heart",
      "Drawing figures large and centered; people standing together, hand in hand"
    ]
  },
  sad: {
    title: "These may be the reasons:",
    reasons: [
      "Extensive use of cold colors such as blue, gray and purple",
      "Slow and pale lines, drawings made with low energy",
      "Sad facial expressions (downturned mouth, drooping eyebrows, tears)",
      "Drawing figures alone or far from each other, leaving too much space on the page"
    ]
  }
};

const handleAnalyze = async () => {
  if (!image) {
    setResult({ emotion: null, message: "Please upload an image." });
    return;
  }

  const formData = new FormData();
  formData.append("file", image);

  try {
    const response = await axios.post('http://localhost:5000/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    console.log(response.data);

    if (response.data.emotion) {
      const emotion = response.data.emotion.toLowerCase();
      
      setResult({
        emotion: emotion,
        message: `Predicted Emotion: ${response.data.emotion}`,
        customMessage: emotionMessages[emotion] || {
          title: "Emotion analysis",
          reasons: ["There is no specific explanation for this emotion."]
        }
      });
    } else {
      setResult({ emotion: null, message: "Emotion prediction was not successful.", customMessage: null });
    }
  } catch (error) {
    console.error('An error occurred during image analysis:', error);
    let errorMessage = "An unexpected error occurred.";
    if (error.response) {
      errorMessage = `Error: ${error.response.data.error}`;
    } else if (error.request) {
      errorMessage = "Failed to connect to server. Please try again.";
    }
    setResult({ emotion: null, message: errorMessage, customMessage: null });
  }
};

  return (
  <Container className="d-flex justify-content-center align-items-center vh-100">
    <Card className="p-4 shadow-lg hover:shadow-2xl transition duration-300 ease-in-out" style={{ maxWidth: '500px', width: '100%' }}>

      <CardBody>
        <CardTitle tag="h1" className="text-center animate__animated animate__fadeIn animate__delay-1s">
          Emotions of Drawings
        </CardTitle>

        <input
          type="file"
          accept="image/*"
          onChange={handleImageChange}
          className="form-control mb-4"
        />
        {image && <img src={URL.createObjectURL(image)} alt="Uploaded" className="img-fluid mb-4 mx-auto" />}
        <Button onClick={handleAnalyze} color="primary" block className="transition duration-300 transform hover:scale-105">
          <i className="fa fa-magic"></i> Analyze
        </Button>
        {result && (
          <div className="mt-4 text-center">
            <p>{result.message}</p>
            {result.customMessage && (
              <div className="mt-3 p-3 bg-light rounded text-start">
                <p>{result.customMessage.title}</p>
                <ul className="pl-3">
                  {result.customMessage.reasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </CardBody>
    </Card>
  </Container>
);
}

export default App;