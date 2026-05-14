import React, { useMemo, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'

function DeviceCube({ device, index, total }) {
  const [hovered, setHovered] = useState(false)
  const angle = (index / Math.max(total, 1)) * Math.PI * 2
  const radius = 4.5
  const position = [Math.cos(angle) * radius, Math.sin(angle) * radius, 0]
  const attacked = device.logs.some((x) => x.prediction === 'ATTACK')

  return (
    <group>
      <mesh position={position} onPointerOver={() => setHovered(true)} onPointerOut={() => setHovered(false)}>
        <boxGeometry args={[0.8, 0.8, 0.8]} />
        <meshStandardMaterial emissive={attacked ? '#ff002b' : '#00ff9f'} color={attacked ? '#7a0017' : '#003d2f'} />
      </mesh>
      <line>
        <bufferGeometry attach="geometry" />
      </line>
      {hovered && (
        <Html position={[position[0], position[1] + 1, position[2]]}>
          <div className="rounded-xl border border-cyan-400/30 bg-slate-900/90 p-3 text-xs text-cyan-200 shadow-xl">
            <p className="font-semibold">{device.ip}</p>
            {device.logs.map((log, i) => (
              <p key={i}>{log.timestamp} - {log.prediction}</p>
            ))}
          </div>
        </Html>
      )}
    </group>
  )
}

export default function NetworkScene({ devices }) {
  const list = useMemo(() => Object.values(devices), [devices])

  return (
    <div className="h-[520px] w-full rounded-2xl border border-cyan-500/30 bg-slate-950/50 backdrop-blur">
      <Canvas camera={{ position: [0, 0, 12] }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[0, 0, 8]} intensity={2} color="#00f6ff" />
        <mesh>
          <sphereGeometry args={[1, 32, 32]} />
          <meshStandardMaterial color="#1d4ed8" emissive="#00d9ff" emissiveIntensity={1} />
        </mesh>
        {list.map((d, idx) => <DeviceCube key={d.ip} device={d} index={idx} total={list.length} />)}
        <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={0.7} />
      </Canvas>
    </div>
  )
}
