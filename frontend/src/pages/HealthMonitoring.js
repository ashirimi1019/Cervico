import React from 'react';
import { motion } from 'framer-motion';

export default function HealthMonitoring() {
  return (
    <div className="min-h-screen bg-white">
      <motion.section 
        className="pt-32 pb-16 px-6"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <div className="max-w-7xl mx-auto">
          <motion.h1 
            className="text-5xl font-bold text-primary mb-8"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            Continuous Health Monitoring
          </motion.h1>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <motion.div 
              className="space-y-6"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <h2 className="text-3xl font-semibold text-gray-800">Real-time Health Insights</h2>
              <p className="text-lg text-gray-600">
                Stay informed about your pregnancy health with our advanced monitoring system.
                Receive real-time alerts, personalized recommendations, and connect with healthcare
                providers when needed.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-primary text-xl mr-2">•</span>
                  <span>24/7 health parameter monitoring</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary text-xl mr-2">•</span>
                  <span>Automated risk assessment and alerts</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary text-xl mr-2">•</span>
                  <span>Direct connection to healthcare providers</span>
                </li>
              </ul>
            </motion.div>

            <motion.div 
              className="relative"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.6 }}
            >
              <div className="aspect-w-16 aspect-h-9 rounded-lg overflow-hidden shadow-2xl">
                <img 
                  src="/images/health-monitoring-system.jpg" 
                  alt="Health Monitoring System"
                  className="object-cover"
                />
              </div>
              <div className="absolute -bottom-6 -right-6 w-32 h-32 bg-primary/10 rounded-full" />
            </motion.div>
          </div>
        </div>
      </motion.section>
    </div>
  );
}
