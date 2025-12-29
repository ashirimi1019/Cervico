import React from "react";
import { Link } from "react-router-dom";

const Features = () => {
  return (
    <section id="features" className="py-16 px-6 bg-white text-center">
      <h2 className="text-4xl font-bold text-blue-600">Key Features</h2>
      <div className="mt-8 grid md:grid-cols-3 gap-6">
        <div className="p-6 bg-gray-100 rounded-lg shadow-md">
          <h3 className="text-xl font-bold text-gray-800">AI-Powered Diagnostics</h3>
          <p className="text-gray-600">Analyze patient symptoms with advanced AI models.</p>
          <Link to="/ultrasound-analysis" className="text-primary hover:text-primary-dark font-medium">Learn More →</Link>
        </div>
        <div className="p-6 bg-gray-100 rounded-lg shadow-md">
          <h3 className="text-xl font-bold text-gray-800">Medication Alerts</h3>
          <p className="text-gray-600">Get reminders for prescriptions and dosages.</p>
          <Link to="/health-tracking" className="text-primary hover:text-primary-dark font-medium">Learn More →</Link>
        </div>
        <div className="p-6 bg-gray-100 rounded-lg shadow-md">
          <h3 className="text-xl font-bold text-gray-800">Health Monitoring</h3>
          <p className="text-gray-600">Track vital signs and receive health insights.</p>
          <Link to="/health-monitoring" className="text-primary hover:text-primary-dark font-medium">Learn More →</Link>
        </div>
      </div>
    </section>
  );
};

export default Features;
