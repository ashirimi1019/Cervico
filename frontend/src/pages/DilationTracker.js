import { motion } from "framer-motion";

export default function DilationTracker() {
  return (
    <div className="min-h-screen pt-20 px-6 bg-gray-50">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold text-primary mb-8">Dilation Tracker</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-white p-6 rounded-lg shadow-soft">
            <h2 className="text-2xl font-bold text-primary mb-4">Current Status</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-gray-700 mb-2">Date</label>
                <input
                  type="date"
                  className="w-full p-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-gray-700 mb-2">Dilation (cm)</label>
                <input
                  type="number"
                  min="0"
                  max="10"
                  step="0.1"
                  className="w-full p-2 border rounded-lg"
                />
              </div>
              <div>
                <label className="block text-gray-700 mb-2">Notes</label>
                <textarea
                  className="w-full p-2 border rounded-lg"
                  rows="4"
                ></textarea>
              </div>
              <button className="btn-primary w-full">
                Save Measurement
              </button>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-soft">
            <h2 className="text-2xl font-bold text-primary mb-4">Progress Chart</h2>
            <div className="h-64 bg-gray-100 rounded-lg flex items-center justify-center">
              Chart Placeholder
            </div>
          </div>
        </div>
      </div>
    </div>
  );
} 