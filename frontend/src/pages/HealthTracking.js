import React from 'react';
import { motion } from 'framer-motion';

export default function HealthTracking() {
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
            Comprehensive Health Tracking
          </motion.h1>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            <motion.div 
              className="space-y-6"
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              <h2 className="text-3xl font-semibold text-gray-800">Monitor Your Pregnancy Journey</h2>
              <p className="text-lg text-gray-600">
                Keep track of all important aspects of your pregnancy with our comprehensive health
                tracking system. Monitor cervical changes, record symptoms, and track important
                milestones throughout your journey.
              </p>
              <ul className="space-y-4">
                <li className="flex items-start">
                  <span className="text-primary text-xl mr-2">•</span>
                  <span>Cervical length tracking over time</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary text-xl mr-2">•</span>
                  <span>Symptom logging and pattern recognition</span>
                </li>
                <li className="flex items-start">
                  <span className="text-primary text-xl mr-2">•</span>
                  <span>Customizable health metrics and goals</span>
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
                  src="/images/health-tracking-dashboard.jpg" 
                  alt="Health Tracking Dashboard"
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
