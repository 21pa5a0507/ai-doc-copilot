import axios from "axios";

const API = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

export const askQuestion = async (question, source = "default", history = []) => {
  const res = await axios.post(`${API}/ask`, { question, source, history });
  return res.data;
};
