import React from "react";

const Contact = () => {
  return (
    <section id="contact" className="py-16 px-6 bg-gray-100 text-center">
      <h2 className="text-4xl font-bold text-blue-600">Contact Us</h2>
      <form className="mt-8 max-w-lg mx-auto bg-white p-6 rounded-lg shadow-md">
        <input type="text" placeholder="Your Name" className="w-full p-3 border rounded mb-4" />
        <input type="email" placeholder="Your Email" className="w-full p-3 border rounded mb-4" />
        <textarea placeholder="Your Message" className="w-full p-3 border rounded mb-4"></textarea>
        <button className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700">Send Message</button>
      </form>
    </section>
  );
};

export default Contact;
