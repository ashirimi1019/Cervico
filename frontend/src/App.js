import React from 'react';
import { motion } from "framer-motion";
import { useState } from "react";
import { Routes, Route, Link, useNavigate } from "react-router-dom";
import UltrasoundUpload from "./pages/UltrasoundUpload";
import DilationTracker from "./pages/DilationTracker";
import UltrasoundAnalysis from "./pages/UltrasoundAnalysis";
import HealthTracking from "./pages/HealthTracking";
import HealthMonitoring from "./pages/HealthMonitoring";
import Community from "./pages/Community";
import Result from './pages/Result';

export default function App() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const mainNavItems = [
    { name: "Home", path: "/" },
    { name: "Upload Ultrasound", path: "/upload-ultrasound" },
    { name: "Dilation Tracker", path: "/dilation-tracker" },
    { name: "Health Tracking", path: "/health-tracking" },
    { name: "Community", path: "/community" }
  ];

  return (
    <div className="min-h-screen flex flex-col bg-fixed bg-cover bg-center font-tinos"
      style={{ backgroundImage: "url('/images/backdrop.png')" }}>
        
        {/* Navigation Bar */}
        <motion.header 
          className="w-full fixed top-0 bg-white py-4 px-6 z-50 shadow-md"
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            {/* Logo Section */}
            <div className="flex items-center gap-3">
              <img src="/images/logo.png" alt="Cervico Logo" className="h-10 w-auto" />
              <h1 className="text-2xl font-bold text-primary">Cervico</h1>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden md:block">
              <ul className="flex gap-8">
                {mainNavItems.map((item) => (
                  <li key={item.name}>
                    <Link
                      to={item.path}
                      className="text-secondary hover:text-primary transition-colors duration-300"
                    >
                      {item.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>

            {/* Mobile Menu Button */}
            <button 
              className="md:hidden text-secondary"
              onClick={() => setIsMenuOpen(!isMenuOpen)}
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d={isMenuOpen ? "M6 18L18 6M6 6l12 12" : "M4 6h16M4 12h16M4 18h16"} />
              </svg>
            </button>
          </div>

          {/* Mobile Navigation */}
          <motion.nav 
            className={`md:hidden ${isMenuOpen ? 'block' : 'hidden'} absolute top-full left-0 right-0 bg-white shadow-soft`}
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: isMenuOpen ? 1 : 0, y: isMenuOpen ? 0 : -20 }}
          >
            <ul className="px-6 py-4 space-y-4">
              <li>
                <Link to="/" onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })} className="nav-link">Home</Link>
              </li>
              <li>
                <Link to="/upload-ultrasound" className="nav-link">Upload Ultrasound</Link>
              </li>
              <li>
                <Link to="/dilation-tracker" className="nav-link">Dilation Tracker</Link>
              </li>
              <li>
                <Link to="/health-tracking" className="nav-link">Health Tracking</Link>
              </li>
              <li>
                <Link to="/community" className="nav-link">Community</Link>
              </li>
            </ul>
          </motion.nav>
        </motion.header>

        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload-ultrasound" element={<UltrasoundUpload />} />
          <Route path="/ultrasound-analysis" element={<UltrasoundAnalysis />} />
          <Route path="/dilation-tracker" element={<DilationTracker />} />
          <Route path="/health-tracking" element={<HealthTracking />} />
          <Route path="/health-monitoring" element={<HealthMonitoring />} />
          <Route path="/community" element={<Community />} />
          <Route path="/result" element={<Result />} />
        </Routes>

        {/* Footer */}
        <footer className="bg-secondary text-white py-8 px-6 mt-auto">
          <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <h3 className="text-xl font-bold mb-4">Cervico</h3>
              <p className="text-gray-300">Advanced AI-powered cervical health monitoring</p>
            </div>
            <div>
              <h4 className="font-bold mb-4">Features</h4>
              <ul className="space-y-2">
                <li>AI Analysis</li>
                <li>Dilation Tracking</li>
                <li>Health Monitoring</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold mb-4">Resources</h4>
              <ul className="space-y-2">
                <li>Community</li>
                <li>Contact Support</li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold mb-4">Legal</h4>
              <ul className="space-y-2">
                <li>Privacy Policy</li>
                <li>Terms of Service</li>
                <li>Data Security</li>
              </ul>
            </div>
          </div>
          <div className="max-w-7xl mx-auto mt-8 pt-8 border-t border-gray-700 text-center text-gray-300">
            <p>&copy; 2025 Cervico. All Rights Reserved.</p>
          </div>
        </footer>
      </div>
  );
}

function HomePage() {
  const navigate = useNavigate();

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const careCategories = [
    {
      id: 'pregnancy-guide',
      title: "Late-Stage Pregnancy Guide",
      icon: "",
      description: "Essential tips and guidance for navigating the final weeks of pregnancy.",
      content: `The final weeks of pregnancy can be a mix of excitement and exhaustion as your body prepares for labor. Our Late-Stage Pregnancy Guide offers essential insights to help you stay comfortable, informed, and prepared.

What to Expect in Late Pregnancy
• Baby's Positioning: Your baby will start settling into the head-down position, preparing for birth
• Increased Discomfort: Back pain, pelvic pressure, swollen feet, and shortness of breath are common
• Fatigue & Sleep Struggles: Your growing belly may make it difficult to sleep—try side sleeping with pillows for support
• Braxton Hicks vs. Real Contractions: Learn how to distinguish between practice contractions and actual labor
• Emotional Changes: You may feel anxious, excited, or even a little impatient—this is completely normal

Final Preparations
• Packing Your Hospital Bag: Essentials include comfy clothes, baby items, documents, and personal care products
• Preparing Your Home: Set up the nursery, stock up on postpartum care supplies, and organize baby essentials
• Knowing When to Go to the Hospital: Learn the signs of active labor, water breaking, and when to contact your healthcare provider

Being well-informed helps you feel more confident and in control as labor approaches.`,
      images: {
        main: "/images/pregnancy-guide-main.jpg",
      },
      theme: {
        background: "linear-gradient(45deg, #f3e7ff 0%, #ffffff 100%)",
        accent: "rgba(147, 51, 234, 0.1)"
      }
    },
    {
      id: 'community-support',
      title: "Community Support",
      icon: "",
      description: "Connect with other expectant mothers and find support during your pregnancy journey.",
      content: `Having a strong support system is crucial during late pregnancy. Our community offers various ways to connect and share experiences.

Support Resources
• Peer Support Groups: Connect with other expectant mothers in your area
• Partner Resources: Information to help your partner understand and support you
• Online Forums: Safe spaces to discuss concerns and share experiences
• Local Support Services: Access to pregnancy support services in your community

Expert Q&A
• Common Late Pregnancy Concerns: Expert answers to frequently asked questions
• Medical Guidance: When to contact your healthcare provider
• Birth Plan Support: Help with creating and understanding birth plans

Emergency Resources
• 24/7 Support Lines: Access to medical professionals when needed
• Warning Signs: Know when to seek immediate medical attention
• Hospital Information: Important contacts and locations for emergencies`,
      images: {
        main: "/images/community-main.jpg",
      },
      theme: {
        background: "linear-gradient(45deg, #e7f5ff 0%, #ffffff 100%)",
        accent: "rgba(14, 165, 233, 0.1)"
      }
    },
    {
      id: 'expert-qa',
      title: "Expert Q&A",
      icon: "",
      description: "Get rapid medical consultations from professionals when you need them.",
      content: `Have pregnancy concerns but don't want to wait for your next doctor's appointment? Our Expert Q&A section provides direct access to medical professionals, ensuring you receive reliable, quick answers when you need them.

Common Topics Covered
• Signs of Labor: Is it time to go to the hospital? Learn how to recognize true labor
• Unusual Symptoms: Worried about pain, swelling, or sudden changes? Get expert advice fast
• Medications & Safe Remedies: Know which medications and natural remedies are safe for late pregnancy
• Labor Pain Management: Explore different options, from breathing techniques to epidurals

When it comes to pregnancy, peace of mind is priceless—get trustworthy guidance, fast.`,
      images: {
        main: "/images/expert-qa-main.jpg",
      },
      theme: {
        background: "linear-gradient(45deg, #ecfdf5 0%, #ffffff 100%)",
        accent: "rgba(16, 185, 129, 0.1)"
      }
    },
    {
      id: 'emergency',
      title: "Emergency Resources",
      icon: "",
      description: "Access critical emergency response tools for labor and late pregnancy.",
      content: `No one expects emergencies, but being prepared can make all the difference. Our Emergency Resources section helps you act quickly and confidently if unexpected situations arise.

Key Resources Available
• When to Seek Immediate Medical Help: Know the warning signs of serious conditions like pre-eclampsia, placental abruption, or preterm labor
• Hospital & Birthing Center Information: Locate the nearest maternity facilities and plan your route
• Emergency Contacts & Hotlines: Have important phone numbers ready for medical emergencies
• What to Do If Labor Starts Unexpectedly: Learn what steps to take if labor begins before you reach the hospital

With the right preparation, you can handle any situation with confidence and clarity.`,
      images: {
        main: "/images/emergency-main.jpg",
      },
      theme: {
        background: "linear-gradient(45deg, #f0fdf4 0%, #ffffff 100%)",
        accent: "rgba(34, 197, 94, 0.1)"
      }
    },
    {
      id: 'exercise',
      title: "Exercise Plans",
      icon: "",
      description: "Safe, doctor-recommended movements to ease discomfort and prepare for labor.",
      content: `Gentle movement can reduce pregnancy discomfort, improve circulation, and help prepare your body for labor. Our Exercise Plans focus on safe, doctor-approved workouts tailored for the final weeks of pregnancy.

Safe & Effective Exercises for Late Pregnancy
• Pelvic Tilts & Cat-Cow Stretches: Helps with back pain and promotes better posture
• Deep Squats & Hip Openers: Strengthens the pelvic floor and may encourage labor
• Prenatal Yoga & Breathing Exercises: Increases flexibility, reduces stress, and improves labor endurance
• Walking & Light Cardio: Helps baby descend into position and keeps your energy up

Benefits of Staying Active
• Reduces swelling, back pain, and tension
• Improves flexibility and stamina for labor
• Helps with better sleep and mood stabilization

Movement is medicine—even simple exercises can make labor easier and boost overall well-being.`,
      images: {
        main: "/images/exercise-main.jpg",
      },
      theme: {
        background: "linear-gradient(45deg, #ecfdf5 0%, #ffffff 100%)",
        accent: "rgba(16, 185, 129, 0.1)"
      }
    },
    {
      id: 'nutrition',
      title: "Nutrition Advice",
      icon: "",
      description: "Expert-backed diet tips to support both mom and baby in the final stretch.",
      content: `Eating the right foods in late pregnancy can boost energy, strengthen immunity, and prepare your body for labor. Our Nutrition Advice section provides tailored guidance on what to eat and avoid in the final trimester.

Best Foods for Late Pregnancy
• Iron-Rich Foods: Spinach, lean meats, and legumes prevent anemia and boost oxygen flow to your baby
• Protein & Healthy Fats: Eggs, nuts, and fish support brain development and energy levels
• Fiber & Hydration: Whole grains, fruits, and plenty of water help with digestion and prevent constipation
• Labor-Boosting Foods: Dates, pineapple, and spicy foods may help naturally encourage contractions

Foods to Avoid
• Excess Caffeine & Sugar: Can cause energy crashes and affect fetal development
• Raw or Undercooked Foods: Avoid raw fish, deli meats, and unpasteurized dairy to prevent infections
• Overly Salty & Processed Foods: Can contribute to bloating and high blood pressure

By making smart food choices, you can nourish both yourself and your baby in the final stretch.`,
      images: {
        main: "/images/nutrition-main.jpg",
      },
      theme: {
        background: "linear-gradient(45deg, #f0fdf4 0%, #ffffff 100%)",
        accent: "rgba(34, 197, 94, 0.1)"
      }
    }
  ];

  return (
    <div>
      {/* Hero Section */}
      <motion.section 
        className="pt-32 pb-16 px-6 min-h-screen flex items-center justify-center bg-cover bg-center"
        style={{
          backgroundImage: "url('/images/backdrop.png')",
          backgroundPosition: 'center 30%'
        }}
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center">
            <motion.h1 
              className="text-6xl font-bold text-primary mb-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              Cervical Health Monitoring
            </motion.h1>
            <motion.p 
              className="text-2xl text-gray-700 mb-12 max-w-3xl mx-auto"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              Advanced AI-powered ultrasound analysis for accurate cervical measurements
            </motion.p>
            <motion.div 
              className="flex flex-col sm:flex-row justify-center gap-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <button 
                className="btn-primary"
                onClick={() => navigate('/upload-ultrasound')}
              >
                Upload Ultrasound
              </button>
              <button 
                className="btn-secondary"
                onClick={() => navigate('/dilation-tracker')}
              >
                Track Dilation
              </button>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Features Section */}
      <section className="py-16 bg-white">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-4xl font-bold text-primary text-center mb-12">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              {
                title: "AI Analysis",
                description: "Advanced ultrasound analysis using state-of-the-art AI technology",
                link: "/ultrasound-analysis"
              },
              {
                title: "Health Tracking",
                description: "Comprehensive tracking of cervical health measurements",
                link: "/health-tracking"
              },
              {
                title: "Continuous Monitoring",
                description: "Real-time monitoring and alerts for your pregnancy journey",
                link: "/health-monitoring"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="bg-white p-6 rounded-lg"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.2 }}
              >
                <h3 className="text-xl font-semibold mb-4">{feature.title}</h3>
                <p className="text-gray-600 mb-4">{feature.description}</p>
                <Link to={feature.link} className="text-primary hover:text-primary-dark font-medium">
                  Learn More →
                </Link>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Comprehensive Care Section */}
      <section id="comprehensive-care" className="py-16 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-4xl font-bold text-primary text-center mb-12">Comprehensive Care</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {careCategories.map((category, index) => (
              <motion.div 
                key={index} 
                className="bg-white rounded-xl p-8"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
              >
                <h3 className="text-2xl font-semibold mb-3 text-primary">{category.title}</h3>
                <p className="text-gray-600 mb-4 leading-relaxed">{category.description}</p>
                <button
                  onClick={() => scrollToSection(category.id)}
                  className="text-primary hover:text-primary-dark font-medium inline-flex items-center group"
                >
                  <span>Explore</span>
                  <svg 
                    className="w-4 h-4 ml-2 transform group-hover:translate-x-1 transition-transform duration-200" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Detailed Care Category Sections */}
      <div className="bg-white">
        {careCategories.map((category, index) => (
          <motion.section
            key={category.id}
            id={category.id}
            className="min-h-screen relative overflow-hidden py-20"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            {/* Section Background */}
            <div 
              className="absolute inset-0 pointer-events-none opacity-50"
              style={{
                background: category.theme.background
              }}
            />

            {/* Decorative Background Pattern */}
            <div 
              className="absolute inset-0 pointer-events-none"
              style={{
                backgroundImage: `url(${category.images.decorative})`,
                backgroundSize: '400px 400px',
                backgroundRepeat: 'repeat',
                opacity: 0.1
              }}
            />

            {/* Decorative Shapes */}
            <div 
              className="absolute inset-0 pointer-events-none overflow-hidden"
              aria-hidden="true"
            >
              <div 
                className="absolute -top-1/4 -right-1/4 w-1/2 h-1/2 rounded-full blur-3xl"
                style={{ background: category.theme.accent }} 
              />
              <div 
                className="absolute -bottom-1/4 -left-1/4 w-1/2 h-1/2 rounded-full blur-3xl"
                style={{ background: category.theme.accent }} 
              />
            </div>

            {/* Main Content */}
            <div className="max-w-7xl mx-auto px-6 relative">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
                {/* Text Content */}
                <div className="space-y-8 bg-white/80 backdrop-blur-sm rounded-2xl p-8">
                  <h2 className="text-4xl font-bold text-primary mb-6">{category.title}</h2>
                  <div className="prose prose-lg max-w-none">
                    {category.content.split('\n\n').map((paragraph, idx) => (
                      <div key={idx} className="mb-6">
                        {paragraph.startsWith('•') ? (
                          <ul className="list-disc list-inside space-y-2">
                            {paragraph.split('\n').map((item, i) => (
                              <li key={i} className="text-gray-700 leading-relaxed">
                                {item.replace('•', '').trim()}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-gray-700 leading-relaxed">{paragraph}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
                
                {/* Image Content */}
                <div className="relative">
                  <div className="aspect-w-4 aspect-h-3 rounded-2xl overflow-hidden">
                    <img 
                      src={category.images.main} 
                      alt={category.title}
                      className="object-cover w-full h-full"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Navigation Dots */}
            <div className="fixed right-8 top-1/2 transform -translate-y-1/2 space-y-4 z-50 hidden lg:block">
              {careCategories.map((cat, idx) => (
                <button
                  key={cat.id}
                  onClick={() => scrollToSection(cat.id)}
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    cat.id === category.id ? 'bg-primary scale-150' : 'bg-gray-300 hover:bg-primary/50'
                  }`}
                  aria-label={`Go to ${cat.title}`}
                />
              ))}
            </div>
          </motion.section>
        ))}
      </div>
    </div>
  );
}
