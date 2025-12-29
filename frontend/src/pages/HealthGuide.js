import React from 'react';
import { motion } from 'framer-motion';

export default function HealthGuide() {
  return (
    <div className="pt-24 px-6">
      <motion.div 
        className="max-w-7xl mx-auto"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-bold text-primary mb-8">Health Guide</h1>
        <div className="grid md:grid-cols-2 gap-8">
          {/* Content will be added here */}
          <div className="bg-white rounded-lg p-6 shadow-md">
            <h2 className="text-2xl font-bold text-primary mb-4">Coming Soon</h2>
            <p className="text-gray-600">
              Our comprehensive health guide is currently under development. 
              Check back soon for detailed information about cervical health monitoring.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
