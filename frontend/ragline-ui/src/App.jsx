import { useEffect, useState } from "react";
import ChatBox from "./components/ChatBox";
import SourcesPanel from "./components/SourcesPanel";

const THEME_STORAGE_KEY = "guide-layer-theme";

function getChatPreview(chat) {
  const firstUserMessage = chat.find((message) => message.role === "user")?.text || "Untitled chat";
  return firstUserMessage.length > 44
    ? `${firstUserMessage.slice(0, 44).trim()}...`
    : firstUserMessage;
}

function App() {
  const [sources, setSources] = useState([]);
  const [history, setHistory] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [messages, setMessages] = useState([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isLoadedChat, setIsLoadedChat] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem(THEME_STORAGE_KEY) || "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const saveCurrentChat = () => {
    if (!isLoadedChat && messages.length > 0) {
      setHistory((prev) => [...prev, messages]);
    }
  };

  const handleNewChat = () => {
    saveCurrentChat();
    setMessages([]);
    setActiveChat(null);
    setSources([]);
    setSidebarOpen(false);
    setIsLoadedChat(false);
  };

  const handleSelectChat = (chat) => {
    saveCurrentChat();
    setMessages(chat);
    setActiveChat(chat);
    setSidebarOpen(false);
    setIsLoadedChat(true);
  };

  return (
    <div className="app-shell">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute left-[-10rem] top-[-8rem] h-64 w-64 rounded-full bg-[var(--glow-primary)] blur-3xl opacity-25" />
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-[var(--border-strong)] to-transparent opacity-70" />
      </div>

      {sidebarOpen && (
        <button
          type="button"
          aria-label="Close sidebar"
          className="fixed inset-0 z-30 bg-black/50 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="relative z-10 flex h-screen overflow-hidden">
        <aside
          className={`fixed inset-y-0 left-0 z-40 w-[17rem] shrink-0 transform border-r border-[var(--border)] bg-[var(--sidebar-bg)]/96 px-4 py-5 backdrop-blur-xl transition-transform duration-300 md:static md:h-screen md:translate-x-0 ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="flex h-full flex-col gap-5">
            <div className="flex items-center gap-3 px-1">
              <div className="brand-mark">HP</div>
              <div>
                <p className="font-display text-lg font-semibold text-[var(--text-primary)]">
                  HexPilot AI
                </p>
                <p className="text-xs text-[var(--text-muted)]">Workspace</p>
              </div>
            </div>

            <button type="button" onClick={handleNewChat} className="primary-button w-full justify-center">
              Start New Chat
            </button>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">Recent Sessions</p>
                <p className="text-xs text-[var(--text-muted)]">
                  {history.length} saved conversation{history.length === 1 ? "" : "s"}
                </p>
              </div>
            </div>

            <div className="chat-scroll flex-1 space-y-3 overflow-y-auto pr-1">
              {history.length === 0 && (
                <div className="panel-soft p-4">
                  <p className="text-sm font-medium text-[var(--text-primary)]">No saved chats yet</p>
                  <p className="mt-2 text-sm leading-6 text-[var(--text-muted)]">
                    Start a conversation and it will appear here for quick access.
                  </p>
                </div>
              )}

              {history.map((chat, index) => {
                const isActive = activeChat === chat;
                return (
                  <button
                    key={`${getChatPreview(chat)}-${index}`}
                    type="button"
                    onClick={() => handleSelectChat(chat)}
                    className={`sidebar-item ${isActive ? "sidebar-item-active" : ""}`}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold leading-6 text-[var(--text-primary)]">
                          {getChatPreview(chat)}
                        </p>
                        <p className="mt-2 text-xs text-[var(--text-muted)]">
                          {chat.length} message{chat.length === 1 ? "" : "s"}
                        </p>
                      </div>
                      <span className="mt-1 h-2.5 w-2.5 rounded-full bg-[var(--accent-soft)]" />
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </aside>

        <main className="flex min-w-0 min-h-0 flex-1 flex-col overflow-hidden">
          <header className="border-b border-[var(--border)] bg-[var(--surface-elevated)]/92 px-4 py-4 backdrop-blur-xl sm:px-6">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  aria-label="Open sidebar"
                  onClick={() => setSidebarOpen((prev) => !prev)}
                  className="icon-button md:hidden"
                >
                  <span className="text-base">≡</span>
                </button>

                <div>
                  <div className="flex items-center gap-3">
                    <div className="brand-mark brand-mark-small">HP</div>
                    <div>
                      <p className="font-display text-xl font-semibold tracking-tight text-[var(--text-primary)]">
                        HexPilot AI
                      </p>
                      <p className="text-sm text-[var(--text-muted)]">
                        Search across Hexnode docs and Keka policies
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setTheme((prev) => (prev === "dark" ? "light" : "dark"))}
                  className="theme-button"
                >
                  <span className="theme-button-indicator" />
                  {theme === "dark" ? "Switch to Light" : "Switch to Dark"}
                </button>
              </div>
            </div>
          </header>

          <div className="flex min-h-0 flex-1 overflow-hidden">
            <section className="flex min-w-0 min-h-0 flex-1 flex-col px-3 py-3 sm:px-4 sm:py-4">
              <div className="panel flex min-h-0 flex-1 overflow-hidden">
                <ChatBox
                  setSources={setSources}
                  messages={messages}
                  setMessages={setMessages}
                  setIsLoadedChat={setIsLoadedChat}
                />
              </div>
            </section>

            <section className="hidden min-h-0 w-[21rem] shrink-0 py-4 pr-4 xl:block">
              <div className="panel h-full overflow-hidden">
                <SourcesPanel sources={sources} />
              </div>
            </section>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
