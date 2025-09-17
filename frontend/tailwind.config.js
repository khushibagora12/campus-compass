/** @type {import('tailwindcss').Config} */
export default {
  content: [
     "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        poppins: ['Poppins', 'sans-serif'],
        italiana: ['Italiana', 'sans-serif'],
        bebas: ['Bebas Neue', 'sans-serif'],
        bevan: ['Bevan', 'sans-serif'],
        fascinate: ['Fascinate', 'sans-serif'],
        frederica: ['Fredericka the Great', 'sans-serif'],
        rye: ['Rye', 'sans-serif'],
        bungee: ['Bungee Shade', 'sans-serif']
      },
      animation: {
        'star-movement-bottom': 'star-movement-bottom linear infinite alternate',
        'star-movement-top': 'star-movement-top linear infinite alternate',
      },
      keyframes: {
        'star-movement-bottom': {
          '0%': { transform: 'translate(0%, 0%)', opacity: '1' },
          '100%': { transform: 'translate(-100%, 0%)', opacity: '0' },
        },
        'star-movement-top': {
          '0%': { transform: 'translate(0%, 0%)', opacity: '1' },
          '100%': { transform: 'translate(100%, 0%)', opacity: '0' },
        },
      },
    },
  },
  plugins: [],
}

