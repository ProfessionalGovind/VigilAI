import React, { useState, useEffect } from 'react';

// This is a simple React functional component
function App() {
  // State to hold the message we get from the FastAPI backend
  const [apiMessage, setApiMessage] = useState("Connecting to API...");
  const [modelsLoaded, setModelsLoaded] = useState(0);

  // useEffect runs when the component first loads (mounts)
  useEffect(() => {
    // This is where we call the FastAPI test endpoint (http://localhost:8000/)
    fetch('http://localhost:8000/')
      .then(res => res.json())
      .then(data => {
        // Update the state with the successful data from the backend
        setApiMessage(data.message);
        setModelsLoaded(data.models_loaded);
      })
      .catch(error => {
        // If the connection fails, show an error message
        setApiMessage("Error: Could not connect to backend API!");
        console.error("Fetch error:", error);
      });
  }, []); // The empty array [] means this effect runs only once

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>VigilAI Dashboard Prototype</h1>
      <p>Status: <span style={{ color: apiMessage.includes("Error") ? 'red' : 'green', fontWeight: 'bold' }}>
          {apiMessage}
        </span>
      </p>
      {modelsLoaded > 0 && (
        <p>
          ✅ **Backend Ready:** Successfully loaded {modelsLoaded} detection models.
        </p>
      )}
      
      <hr/>
      <h2>Full-Stack Architecture Confirmed</h2>
      <p>This page confirms that your **React Frontend** is communicating directly with your **FastAPI Backend**.</p>
    </div>
  );
}

export default App;