import React from "react";

const Navbar = () => {
  return (
    <nav className="bg-white shadow-md py-4 px-6 md:px-12 flex justify-between items-center">
      <h1 className="text-2xl font-bold text-blue-600">Cervico</h1>
      <ul className="hidden md:flex space-x-6">
        <li><a href="#features" className="text-gray-600 hover:text-blue-600">Features</a></li>
        <li><a href="#contact" className="text-gray-600 hover:text-blue-600">Contact</a></li>
      </ul>
      <button className="md:hidden text-blue-600 text-2xl">☰</button>
    </nav>
  );
};

export default Navbar;
