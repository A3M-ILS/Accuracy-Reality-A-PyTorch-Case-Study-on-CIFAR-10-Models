import './globals.css'
import { Inter } from 'next/font/google'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
  title: 'CIFAR10 Comparison | ResNet18 vs ResNet101',
  description: 'Upload an image and see how ResNet18 and ResNet101 compare.',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-black text-slate-100 selection:bg-sky-500/30`}>
        {children}
      </body>
    </html>
  )
}
