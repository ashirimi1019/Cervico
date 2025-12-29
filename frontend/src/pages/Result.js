import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';

function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageData, setImageData] = useState(null);
  const result = location.state?.result;

  useEffect(() => {
    if (!result?.id) {
      setError('No result data available');
      return;
    }

    const fetchUltrasoundData = async () => {
      try {
        setLoading(true);
        const response = await fetch(`http://localhost:8001/ultrasound/${result.id}`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(data.detail || 'Failed to fetch ultrasound data');
        }

        // Set the base64 image data from MongoDB or use the preview URL
        setImageData(data.data?.image ? `data:image/jpeg;base64,${data.data.image}` : result.image_url);
      } catch (err) {
        console.error('Error fetching ultrasound data:', err);
        // Fall back to the preview URL if fetching fails
        setImageData(result.image_url);
      } finally {
        setLoading(false);
      }
    };

    fetchUltrasoundData();
  }, [result?.id, result?.image_url]);

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-xl text-gray-600">No result data available</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-12 px-4">
      <div className="max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-lg shadow-lg p-8"
        >
          <h1 className="text-3xl font-bold text-primary mb-6">Analysis Result</h1>
          
          {loading && (
            <div className="text-center py-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
            </div>
          )}

          {error && (
            <div className="mb-6 p-4 bg-red-50 text-red-600 rounded-lg">
              {error}
            </div>
          )}

          {/* Prediction Results */}
          <div className="mb-8">
            <h2 className="text-xl font-semibold mb-4">Cervical Dilation</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 rounded-lg p-6">
                <p className="text-sm text-gray-500 mb-1">Classification</p>
                <p className="text-2xl font-bold text-primary">
                  {result.class_prediction.toFixed(1)} cm
                </p>
              </div>
              <div className="bg-gray-50 rounded-lg p-6">
                <p className="text-sm text-gray-500 mb-1">Precise Measurement</p>
                <p className="text-2xl font-bold text-primary">
                  {result.precise_prediction.toFixed(2)} cm
                </p>
              </div>
            </div>
          </div>

          {/* Image Preview */}
          {imageData && (
            <div className="mb-8">
              <h2 className="text-xl font-semibold mb-4">Analyzed Image</h2>
              <div className="rounded-lg overflow-hidden border border-gray-200">
                <img
                  src={imageData}
                  alt="Analyzed ultrasound"
                  className="w-full h-auto"
                />
              </div>
            </div>
          )}

          {/* Timestamp */}
          <div className="text-sm text-gray-500 mt-8">
            Analysis completed at: {new Date(result.timestamp).toLocaleString()}
          </div>

          {/* Actions */}
          <div className="mt-8 flex justify-end space-x-4">
            <button
              onClick={() => navigate('/upload-ultrasound')}
              className="bg-primary text-white px-6 py-2 rounded-lg hover:bg-primary-dark transition-colors"
            >
              Upload Another Image
            </button>
            <button
              onClick={() => window.print()}
              className="bg-secondary text-white px-6 py-2 rounded-lg hover:bg-secondary-dark transition-colors"
            >
              Print Report
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

export default Result;
