import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000'
})

export const checkIpPrediction = async (ipAddress) => {
  const { data } = await api.post('/api/predict', { ip_address: ipAddress })
  return data
}
