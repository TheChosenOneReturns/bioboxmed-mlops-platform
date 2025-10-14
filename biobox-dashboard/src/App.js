import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bar } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// URL de nuestra API del backend
const API_URL = 'http://localhost:8000';

function App() {
  const [models, setModels] = useState([]);
  const [feedbackData, setFeedbackData] = useState({ correct: 0, incorrect: 0 });

  // --- FUNCIÓN MODIFICADA PARA OBTENER DATOS REALES ---
  const fetchData = async () => {
    try {
      // 1. Llama al nuevo endpoint /models para obtener la lista de modelos
      const modelsResponse = await axios.get(`${API_URL}/models`);
      setModels(modelsResponse.data);

      // 2. Llama al nuevo endpoint /feedback/summary para el resumen
      const feedbackResponse = await axios.get(`${API_URL}/feedback/summary`);
      setFeedbackData(feedbackResponse.data);

    } catch (error) {
      console.error("Error al obtener los datos del backend:", error);
      // Mantenemos los arrays vacíos si hay un error para no romper la UI
      setModels([]);
      setFeedbackData({ correct: 0, incorrect: 0 });
    }
  };

  useEffect(() => {
    fetchData();
    // Añadimos un intervalo para que el dashboard se refresque solo cada 10 segundos
    const interval = setInterval(fetchData, 10000); 
    return () => clearInterval(interval); // Limpiar el intervalo al desmontar el componente
  }, []);

  // --- Configuración para los gráficos (sin cambios, ahora usarán datos dinámicos) ---
  const feedbackChartData = {
    labels: ['Aciertos', 'Errores'],
    datasets: [
      {
        label: 'Conteo de Feedback',
        data: [feedbackData.correct, feedbackData.incorrect],
        backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(255, 99, 132, 0.6)'],
        borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
        borderWidth: 1,
      },
    ],
  };

  const accuracyChartData = {
    labels: models.filter(m => m.status === 'COMPLETED').map(m => m.model_id.split('_v')[0]),
    datasets: [{
      label: 'Precisión del Modelo (%)',
      data: models.filter(m => m.status === 'COMPLETED').map(m => m.accuracy * 100),
      backgroundColor: 'rgba(54, 162, 235, 0.6)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 1,
    }]
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Rendimiento en Tiempo Real' },
    },
    scales: { y: { beginAtZero: true } }
  };

  return (
    <div className="container">
      <header className="header">
        <h1>Dashboard Dinámico de MLOps - BioboxMed</h1>
      </header>
      
      <div className="grid">
        <div className="card">
          <h3>Modelos Entrenados</h3>
          {models.length > 0 ? (
            <table>
              <thead>
                <tr>
                  <th>ID del Modelo</th>
                  <th>Estado</th>
                  <th>Precisión</th>
                </tr>
              </thead>
              <tbody>
                {models.map((model) => (
                  <tr key={model.model_id}>
                    <td>{model.model_id}</td>
                    <td>
                      <span className={`status status-${model.status.toLowerCase()}`}>
                        {model.status}
                      </span>
                    </td>
                    <td>{model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p>No hay modelos registrados. ¡Entrena uno para empezar!</p>
          )}
        </div>

        <div className="card">
          <h3>Tasa de Acierto vs Error (Feedback)</h3>
          <div className="chart-container">
            <Bar options={chartOptions} data={feedbackChartData} />
          </div>
        </div>

        <div className="card">
          <h3>Precisión por Modelo</h3>
           <div className="chart-container">
            <Bar options={chartOptions} data={accuracyChartData} />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;