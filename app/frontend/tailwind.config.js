/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#F9FAFB', // Light gray background
        surface: '#FFFFFF', // White cards
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6', // Blue tech
          600: '#2563eb',
          700: '#1d4ed8',
        },
        success: {
          500: '#10b981', // Green completion
          600: '#059669',
        },
        danger: {
          50: '#fef2f2',
          100: '#fee2e2',
          500: '#ef4444', // Red defect
          600: '#dc2626',
        },
        warning: {
          500: '#f59e0b', // Orange warning
          600: '#d97706',
        },
        text: {
          main: '#111827',
          muted: '#6b7280',
        },
        border: '#e5e7eb',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      boxShadow: {
        'soft': '0 4px 20px -2px rgba(0, 0, 0, 0.05)',
        'inner-soft': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.02)',
      }
    },
  },
  plugins: [],
}
