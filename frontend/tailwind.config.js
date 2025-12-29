/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,ts,jsx,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#cf8182',
          light: '#f2a8a9',
          dark: '#b15f60'
        },
        secondary: {
          DEFAULT: '#2A3342',
          light: '#404E65'
        }
      },
      fontFamily: {
        tinos: ["Tinos", "serif", "system-ui"],
        sans: ["Inter", "system-ui", "sans-serif"]
      },
      boxShadow: {
        'soft': '0 4px 20px rgba(0, 0, 0, 0.05)',
        'hover': '0 8px 30px rgba(0, 0, 0, 0.12)'
      }
    },
  },
  plugins: [],
}
