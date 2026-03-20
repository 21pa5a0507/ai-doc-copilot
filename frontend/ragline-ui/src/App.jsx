import { useState } from "react";
import ChatBox from "./components/ChatBox";
import SourcesPanel from "./components/SourcesPanel";

function App() {
  const [sources, setSources] = useState([]);
  const [history, setHistory] = useState([]);
  const [activeChat, setActiveChat] = useState(null);

  return (
    <div className="flex h-screen bg-gray-100">

      {/* Sidebar */}
      <div className="w-64 bg-black text-white p-4">
        <h2 className="font-bold mb-4">Chats</h2>

        <button
          onClick={() => setActiveChat(null)}
          className="bg-white text-black w-full p-2 rounded mb-4"
        >
          + New Chat
        </button>

        <div className="space-y-2">
          {history.map((chat, i) => (
            <div
              key={i}
              onClick={() => setActiveChat(chat)}
              className="p-2 bg-gray-800 rounded cursor-pointer hover:bg-gray-700"
            >
              {chat[0]?.text.slice(0, 20)}...
            </div>
          ))}
        </div>
      </div>

      {/* Chat Section */}
      <div className="flex flex-col w-[65%] bg-white shadow h-full overflow-hidden">
        
        <div className="p-4 border-b font-bold text-lg">
          RAGLINE 🔍
        </div>

        <ChatBox
          setSources={setSources}
          history={history}
          setHistory={setHistory}
          activeChat={activeChat}
        />
      </div>

      {/* Sources Panel */}
      <div className="w-[35%] border-l bg-gray-50">
        <SourcesPanel sources={sources} />
      </div>

    </div>
  );
}

export default App;