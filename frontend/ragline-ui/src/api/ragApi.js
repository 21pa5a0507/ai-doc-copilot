import axios from "axios";

const API = "http://localhost:8000";

export const askQuestion = async (question) => {
  const res = await axios.post(`${API}/ask`, { question });
  return res.data;
};