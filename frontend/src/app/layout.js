import './globals.css';

export const metadata = {
  title: 'DataMind | AI Data Science Platform',
  description: 'AI-powered platform for data analysis, cleaning, and ML',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body>{children}</body>
    </html>
  );
}
