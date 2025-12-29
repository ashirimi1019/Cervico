import React from "react";

const Hero = () => {
  return (
    <section className="h-screen flex flex-col justify-center items-center text-center bg-gray-100 px-6">
      <h2 className="text-5xl font-bold text-blue-600">Revolutionizing Healthcare</h2>
      <p className="mt-4 text-lg text-gray-700 max-w-2xl">
        Cervico leverages AI to streamline patient care, medication management, and health monitoring.
      </p>
      <button className="mt-6 px-6 py-3 bg-blue-600 text-white rounded-lg shadow-md hover:bg-blue-700">
        Get Started
      </button>
    </section>
  );
};

export default Hero;
