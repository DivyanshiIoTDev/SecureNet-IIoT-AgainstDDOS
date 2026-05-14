import React, { useState } from 'react'
import { motion } from 'framer-motion'
import NetworkScene from './components/NetworkScene'
import { checkIpPrediction } from './services/api'

export default function App() {
  const [ip, setIp] = useState('')
  const [devices, setDevices] = useState({})
  const [status, setStatus] = useState('Idle')

  const onCheck = async () => {
    if (!ip) return
    setStatus('Checking...')
    try {
      const data = await checkIpPrediction(ip)
      setDevices((prev) => ({ ...prev, [ip]: { ip, logs: data.logs } }))
      setStatus(`${data.prediction} at ${data.timestamp}`)
    } catch (e) {
      setStatus(e?.response?.data?.detail || 'Request failed')
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-black p-6 text-cyan-100">
      <div className="mx-auto max-w-7xl space-y-6">
        <motion.h1 initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center text-3xl font-bold tracking-widest">
          AI SOC DDoS Detection Dashboard
        </motion.h1>
        <NetworkScene devices={devices} />
        <section className="rounded-2xl border border-cyan-400/20 bg-white/5 p-5 backdrop-blur-md">
          <div className="flex flex-col gap-3 md:flex-row">
            <input value={ip} onChange={(e) => setIp(e.target.value)} placeholder="Enter IP address" className="flex-1 rounded-lg border border-cyan-400/50 bg-black/40 p-3 outline-none" />
            <button onClick={onCheck} className="rounded-lg bg-cyan-500 px-6 py-3 font-semibold text-slate-900 transition hover:bg-cyan-300">CHECK</button>
          </div>
          <p className="mt-3 text-sm text-cyan-300">Status: {status}</p>
        </section>
      </div>
    </main>
  )
}
