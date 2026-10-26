import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:4000/api";
const ML_BASE = import.meta.env.VITE_ML_API_URL || "http://localhost:8000";

export const api = axios.create({ baseURL: API_BASE });
export const mlApi = axios.create({ baseURL: ML_BASE });

export const getRlModels = () => api.get("/rl-models").then((r) => r.data.models);
export const getA2CStrategy = () => api.get("/strategy/a2c").then((r) => r.data);
export const getNews = () => api.get("/news").then((r) => r.data.headlines);

export const getLiveStatus = () => mlApi.get("/live/status").then((r) => r.data);
export const getLiveState = () => mlApi.get("/live/state").then((r) => r.data);
