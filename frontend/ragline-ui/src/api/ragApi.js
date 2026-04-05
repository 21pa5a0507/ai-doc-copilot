import axios from "axios";

const API = "http://localhost:8000";

export const askQuestion = async (question, source = "default") => {
  const res = await axios.post(`${API}/ask`, { question, source });
  return res.data;
};