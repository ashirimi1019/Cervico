import React from 'react';
import { motion } from "framer-motion";
import { useLocation, useNavigate } from 'react-router-dom';

export default function UltrasoundAnalysis() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    return (
      <div className="container mx-auto px-4 py-8 mt-16 text-center">
        <h2 className="text-2xl font-bold text-gray-800">No analysis data available</h2>
        <button
          onClick={() => navigate('/upload-ultrasound')}
          className="mt-4 bg-primary text-white px-6 py-2 rounded-lg"
        >
          Upload New Image
        </button>
      </div>
    );
  }

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="container mx-auto px-4 py-8 mt-16">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="max-w-4xl mx-auto"
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-8 text-center">
          Ultrasound Analysis Results
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* AI Predictions */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-white rounded-lg shadow-xl p-6"
          >
            <h3 className="text-xl font-semibold text-gray-800 mb-4">
              AI Analysis
            </h3>
            <div className="space-y-4">
              <div>
                <p className="text-gray-600">Dilation Stage</p>
                <p className="text-2xl font-bold text-primary">
                  {result.class_prediction} cm
                </p>
              </div>
              <div>
                <p className="text-gray-600">Precise Measurement</p>
                <p className="text-2xl font-bold text-primary">
                  {result.precise_prediction} cm
                </p>
              </div>
            </div>
          </motion.div>

          {/* Sensor Data */}
          {result.sensor_data && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="bg-white rounded-lg shadow-xl p-6"
            >
              <h3 className="text-xl font-semibold text-gray-800 mb-4">
                Sensor Readings
              </h3>
              <div className="space-y-4">
                <div>
                  <p className="text-gray-600">Pressure</p>
                  <p className="text-2xl font-bold text-primary">
                    {result.sensor_data.pressure_mmHg} mmHg
                  </p>
                </div>
                <div>
                  <p className="text-gray-600">Stretch</p>
                  <p className="text-2xl font-bold text-primary">
                    {result.sensor_data.stretch_mm} mm
                  </p>
                </div>
                <div>
                  <p className="text-gray-600">Temperature</p>
                  <p className="text-2xl font-bold text-primary">
                    {result.sensor_data.temperature_C}°C
                  </p>
                </div>
                <div>
                  <p className="text-gray-600">Reading Time</p>
                  <p className="text-sm text-gray-500">
                    {formatTimestamp(result.sensor_data.sensor_timestamp)}
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="mt-8 flex justify-center space-x-4">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/upload-ultrasound')}
            className="bg-primary text-white px-6 py-3 rounded-lg font-semibold"
          >
            Upload Another Image
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={() => navigate('/dilation-tracker')}
            className="bg-secondary text-white px-6 py-3 rounded-lg font-semibold"
          >
            View Dilation History
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
