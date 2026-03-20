import { useState, useEffect, useRef } from "react";
import { askQuestion } from "../api/ragApi";

export default function ChatBox({ setSources, history, setHistory, activeChat }) {
useEffect(() => {
  if (activeChat) {
    setMessages(activeChat);
  }
}, [activeChat]);
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const bottomRef = useRef(null);

  // 🔽 Auto scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 🔥 Typing effect function
  const typeMessage = (text, callback) => {
    let index = 0;
    let currentText = "";

    const interval = setInterval(() => {
      currentText += text[index];
      index++;

      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1].text = currentText;
        return updated;
      });

      if (index === text.length) {
        clearInterval(interval);
        callback && callback();
      }
    }, 15); // speed (lower = faster)
  };

  const handleAsk = async () => {
    if (!question) return;

    const newMessages = [...messages, { role: "user", text: question }];
    setMessages(newMessages);
    setQuestion("");
    setLoading(true);

    try {
      const data = await askQuestion(question);

      // Add empty AI message first
      setMessages((prev) => {
        const updated = [...prev, { role: "ai", text: "" }];

        // Start typing AFTER state update
        setTimeout(() => {
          typeMessage(data.answer);
        }, 100);

        return updated;
      });
      setHistory((prev) => [...prev, [...newMessages, { role: "ai", text: data.answer }]]);
      setSources(data.chunks);

    } catch (err) {
      console.error(err);
    }

    setLoading(false);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4 min-h-0">

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`p-3 rounded-xl max-w-xl ${
              msg.role === "user"
                ? "bg-blue-500 text-white ml-auto"
                : "bg-gray-200 text-black"
            }`}
          >
            {msg.text}
          </div>
        ))}

        {loading && (
          <div className="text-gray-500">Thinking...</div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t flex gap-2 bg-white">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          className="flex-1 border p-3 rounded-lg outline-none"
          placeholder="Ask anything from your docs..."
        />
        <button
          onClick={handleAsk}
          className="bg-black text-white px-4 rounded-lg"
        >
          Send
        </button>
      </div>

    </div>
  );
}