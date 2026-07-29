import { useState, useEffect, useRef } from 'react';
import { 
  FiPlus, FiSend, FiFileText, 
  FiLayers, FiDatabase, 
  FiChevronDown, FiChevronRight, FiMenu, FiX, FiCornerDownRight,
  FiSearch, FiCopy, FiVolume2, FiThumbsUp, FiThumbsDown, FiRefreshCw, FiShare2, FiFolder, FiMoreHorizontal
} from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_URL = import.meta.env.VITE_API_URL || 'https://rag-chatbot-6wk8.onrender.com/api';

interface Citation {
  chunk_id: string;
  document_id: string;
  title: string;
  section_path: string;
  score: number;
  text?: string;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  citations?: Citation[];
  query_type?: string;
  provider?: string;
  latency?: number;
}

interface DocumentItem {
  filename: string;
  upload_date: string;
  chunk_count: number;
  strategy: string;
}

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState('');
  const [strategy] = useState<string>('semantic');
  const [provider, setProvider] = useState<string>('groq');
  const [isLoading, setIsLoading] = useState(false);
  const [_, setCitations] = useState<Citation[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [expandedCitation, setExpandedCitation] = useState<string | null>(null);
  const [showDocModal, setShowDocModal] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch (err) {
      console.error('Failed to fetch documents', err);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile) return;
    setUploading(true);
    setUploadStatus('Uploading & Indexing PDF...');
    const formData = new FormData();
    formData.append('file', uploadFile);
    formData.append('strategy', strategy);

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      if (res.ok) {
        setUploadStatus('✅ Indexed successfully!');
        setUploadFile(null);
        fetchDocuments();
        setTimeout(() => setUploadStatus(''), 3000);
      } else {
        const err = await res.json();
        setUploadStatus(`❌ Upload failed: ${err.detail || 'Error'}`);
      }
    } catch (err) {
      setUploadStatus('❌ Connection error to backend.');
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || inputQuery;
    if (!textToSend.trim() || isLoading) return;

    const userMsgId = Date.now().toString();
    const botMsgId = (Date.now() + 1).toString();

    const userMsg: Message = {
      id: userMsgId,
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const botMsgPlaceholder: Message = {
      id: botMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const currentHistory = messages.map(m => ({ role: m.role, content: m.content }));

    setMessages(prev => [...prev, userMsg, botMsgPlaceholder]);
    if (!queryText) setInputQuery('');
    setIsLoading(true);
    setCitations([]);

    try {
      const response = await fetch(`${API_URL}/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: textToSend,
          chunking_strategy: strategy,
          provider: provider,
          history: currentHistory
        })
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let botResponseText = '';
      let receivedCitations: Citation[] = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunkStr = decoder.decode(value, { stream: true });
        const lines = chunkStr.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataContent = line.slice(6);
            if (dataContent.trim() === '[DONE]') {
              break;
            } else if (dataContent.startsWith('[CITATIONS] ')) {
              try {
                const citeJson = dataContent.replace('[CITATIONS] ', '');
                receivedCitations = JSON.parse(citeJson);
                setCitations(receivedCitations);
              } catch (e) {
                console.error('Failed to parse citations', e);
              }
            } else {
              botResponseText += dataContent;
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === botMsgId
                    ? { ...msg, content: botResponseText, citations: receivedCitations }
                    : msg
                )
              );
            }
          }
        }
      }
    } catch (error) {
      console.error('Error during streaming:', error);
      setMessages(prev =>
        prev.map(msg =>
          msg.id === botMsgId
            ? { ...msg, content: '⚠️ Error generating response. Please check backend logs or try again.' }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setCitations([]);
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const promptSuggestions = [
    { title: 'What services does HMS provide?', desc: 'Overview of AI solutions, RAG, Web & Mobile development', icon: '🏢' },
    { title: 'AI & Automation Capabilities', desc: 'AI Agents, Voice Assistants, Workflow Automation & RAG', icon: '🤖' },
    { title: 'Major Company Projects', desc: 'PropWise AI, Revolution Realty Capital, MyPay & Food Connect', icon: '🚀' },
    { title: 'Headquarters & Leadership', desc: 'Founders, location in Udaipur, Rajasthan & global presence', icon: '📍' }
  ];

  return (
    <div className="flex h-screen bg-[#111111] text-[#ececec] overflow-hidden font-sans">
      
      {/* ── LEFT SIDEBAR (Matching Reference UI) ────────────────────────── */}
      <div 
        className={`fixed md:relative z-40 h-full bg-[#171717] border-r border-[#262626] flex flex-col transition-all duration-300 ${
          isSidebarOpen ? 'w-64' : 'w-0 -translate-x-full md:translate-x-0 md:w-0'
        }`}
      >
        {isSidebarOpen && (
          <div className="flex flex-col h-full p-2.5 space-y-3">
            
            {/* Header: Logo & Title */}
            <div className="flex items-center justify-between px-2 pt-1 pb-1">
              <div className="flex items-center gap-2 font-bold text-sm text-white">
                <div className="px-2 py-0.5 rounded-md bg-white text-black font-extrabold text-xs tracking-tighter lowercase font-mono shadow-sm">
                  hms
                </div>
                <span className="tracking-tight">HMS Chatbot</span>
              </div>
              <button 
                onClick={() => setIsSidebarOpen(false)}
                className="p-1 rounded-md text-[#8e8e8e] hover:text-white hover:bg-[#262626] transition"
              >
                <FiX className="w-4 h-4" />
              </button>
            </div>

            {/* New Chat Button */}
            <button
              onClick={startNewChat}
              className="flex items-center gap-2.5 w-full px-3 py-2 rounded-lg border border-[#2c2c2c] bg-[#1f1f1f] hover:bg-[#282828] text-white text-xs font-medium transition shadow-sm"
            >
              <FiPlus className="w-4 h-4 text-white" />
              <span>New Chat</span>
            </button>

            {/* Navigation Quick Links */}
            <div className="space-y-0.5 text-xs text-[#b4b4b4]">
              <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-[#222222] cursor-pointer">
                <FiSearch className="w-3.5 h-3.5" />
                <span>Search</span>
              </div>
              <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-[#222222] cursor-pointer">
                <FiFileText className="w-3.5 h-3.5" />
                <span>Notes</span>
              </div>
              <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg hover:bg-[#222222] cursor-pointer">
                <FiFolder className="w-3.5 h-3.5" />
                <span>Workspace</span>
              </div>
            </div>

            {/* Knowledge Base Status & Fast Ingestion */}
            <div className="bg-[#1f1f1f] rounded-xl p-2.5 border border-[#2c2c2c] space-y-2 text-xs">
              <div className="flex items-center justify-between text-[#b4b4b4]">
                <span className="flex items-center gap-1.5 font-semibold text-white">
                  <FiDatabase className="text-[#10a37f]" /> Vector Index
                </span>
                <span className="text-[10px] text-[#10a37f] bg-[#10a37f]/10 px-1.5 py-0.5 rounded font-mono">
                  {documents.length} Files
                </span>
              </div>
              <button 
                onClick={() => setShowDocModal(true)}
                className="w-full px-2 py-1 rounded bg-[#2a2a2a] hover:bg-[#333333] border border-[#383838] text-[#ececec] flex items-center justify-between transition text-[11px]"
              >
                <span className="truncate">Manage Documents</span>
                <FiChevronRight className="w-3 h-3 text-[#8e8e8e]" />
              </button>
              
              <div className="pt-1 border-t border-[#2c2c2c]">
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="block w-full text-[11px] text-[#b4b4b4] file:mr-2 file:py-0.5 file:px-2 file:rounded file:border-0 file:text-[10px] file:font-semibold file:bg-[#10a37f] file:text-white cursor-pointer"
                />
                {uploadFile && (
                  <button
                    onClick={handleFileUpload}
                    disabled={uploading}
                    className="w-full mt-1.5 py-1 bg-[#10a37f] hover:bg-[#1a7f64] text-white text-[11px] font-medium rounded transition flex items-center justify-center"
                  >
                    {uploading ? 'Indexing...' : 'Upload & Index PDF'}
                  </button>
                )}
                {uploadStatus && (
                  <p className="text-[10px] text-[#10a37f] mt-1 font-mono">{uploadStatus}</p>
                )}
              </div>
            </div>

            {/* Conversation History */}
            <div className="flex-1 overflow-y-auto space-y-3 pt-1 pr-1 text-xs">
              <div>
                <span className="px-3 text-[10px] font-semibold text-[#666666] uppercase tracking-wider">Active Chat</span>
                <div 
                  onClick={startNewChat}
                  className="mt-1 flex items-center justify-between px-3 py-1.5 rounded-lg text-white bg-[#262626] font-medium cursor-pointer"
                >
                  <span className="truncate">{messages.length > 0 ? (messages[0].content.slice(0, 20) + '...') : 'RAG Conversation'}</span>
                  <FiMoreHorizontal className="w-3.5 h-3.5 text-[#8e8e8e]" />
                </div>
              </div>
            </div>

            {/* Bottom Left User Profile: Rajat */}
            <div className="pt-2 border-t border-[#262626] flex items-center gap-2.5 px-2">
              <div className="w-6 h-6 rounded-full bg-[#ea580c] text-white flex items-center justify-center font-bold text-xs">
                R
              </div>
              <span className="text-xs font-medium text-white truncate">Rajat</span>
            </div>

          </div>
        )}
      </div>

      {/* ── MAIN CHAT VIEW AREA ──────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-[#111111]">
        
        {/* Top Header Controls Bar */}
        <div className="h-12 border-b border-[#222222] bg-[#111111]/90 backdrop-blur-md flex items-center justify-between px-4 z-30">
          <div className="flex items-center gap-2">
            {!isSidebarOpen && (
              <button 
                onClick={() => setIsSidebarOpen(true)}
                className="p-1.5 rounded-md text-[#8e8e8e] hover:text-white hover:bg-[#222222] transition"
              >
                <FiMenu className="w-4 h-4" />
              </button>
            )}
            <span className="text-xs font-semibold text-white">RAG Conversation</span>
            <FiMoreHorizontal className="w-3.5 h-3.5 text-[#8e8e8e] cursor-pointer" />
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[10px] bg-[#10a37f]/10 text-[#10a37f] border border-[#10a37f]/30 px-2 py-0.5 rounded-full font-mono">
              ● RAG Active
            </span>
          </div>
        </div>

        {/* Chat Stream Viewport */}
        <div className="flex-1 overflow-y-auto p-4 md:px-12 lg:px-24 space-y-6">
          
          {/* Welcome Screen / Empty Chat */}
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto py-12 text-center">
              <div className="px-3 py-1 rounded-xl bg-[#ffffff] text-black font-extrabold text-lg tracking-tighter lowercase font-mono shadow-lg mb-4">
                hms
              </div>
              <h2 className="text-xl font-bold text-white mb-2">What would you like to know?</h2>
              <p className="text-xs text-[#8e8e8e] max-w-md mb-8">
                Ask questions about your PDF documents or test casual queries powered by Qdrant & Groq.
              </p>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full text-left">
                {promptSuggestions.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(item.title)}
                    className="p-3 rounded-xl bg-[#1a1a1a] border border-[#2c2c2c] hover:border-[#444444] hover:bg-[#222222] transition group shadow-sm flex flex-col justify-between"
                  >
                    <div className="flex items-center gap-2 text-white font-medium text-xs">
                      <span>{item.icon}</span>
                      <span className="group-hover:text-[#10a37f] transition">{item.title}</span>
                    </div>
                    <span className="text-[11px] text-[#777777] mt-1 line-clamp-1">{item.desc}</span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Active Messages List */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-4 max-w-3xl mx-auto ${
                msg.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {/* Assistant Avatar */}
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-white text-black flex items-center justify-center font-extrabold text-[10px] lowercase font-mono shadow-md shrink-0 mt-1">
                  hms
                </div>
              )}

              {/* Message Box */}
              <div className="flex flex-col max-w-[90%] space-y-2">
                
                {/* Header info */}
                <div className="flex items-center gap-2 text-[11px] text-[#666666]">
                  <span className="font-semibold text-[#ececec]">
                    {msg.role === 'user' ? 'You' : `Open-WebUI (${provider === 'gemini' ? 'Gemini 3.0' : 'Groq LLaMA-3.3'})`}
                  </span>
                  <span>•</span>
                  <span>{msg.timestamp}</span>
                </div>

                {/* Message Content Container */}
                <div
                  className={`p-4 rounded-2xl text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-[#222222] text-white border border-[#333333] rounded-tr-none'
                      : 'bg-transparent text-[#f9fafb] rounded-tl-none border-0'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="markdown-content text-[#f9fafb]">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content || '▌'}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>

                {/* Action Bar under Assistant Response (Matching Reference UI) */}
                {msg.role === 'assistant' && msg.content && (
                  <div className="flex items-center gap-3 pt-1 text-[#777777] text-xs px-1">
                    <button onClick={() => copyToClipboard(msg.content, msg.id)} title="Copy Response" className="hover:text-white transition">
                      <FiCopy className="w-3.5 h-3.5" />
                    </button>
                    <button title="Read Aloud" className="hover:text-white transition">
                      <FiVolume2 className="w-3.5 h-3.5" />
                    </button>
                    <button title="Good Response" className="hover:text-white transition">
                      <FiThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button title="Bad Response" className="hover:text-white transition">
                      <FiThumbsDown className="w-3.5 h-3.5" />
                    </button>
                    <button onClick={() => handleSend(messages[messages.length - 2]?.content)} title="Regenerate" className="hover:text-white transition">
                      <FiRefreshCw className="w-3.5 h-3.5" />
                    </button>
                    <button title="Share Response" className="hover:text-white transition">
                      <FiShare2 className="w-3.5 h-3.5" />
                    </button>
                    {copiedId === msg.id && (
                      <span className="text-[10px] text-[#10a37f] font-mono">Copied!</span>
                    )}
                  </div>
                )}

                {/* Open-WebUI Citation Inspector Accordion */}
                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 bg-[#181818] border border-[#2a2a2a] rounded-xl p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-[#10a37f]">
                      <span className="flex items-center gap-1.5">
                        <FiLayers /> Source Inspector ({msg.citations.length} Verified Chunks)
                      </span>
                    </div>

                    <div className="space-y-1.5 pt-1">
                      {msg.citations.map((cite, cIdx) => (
                        <div key={cIdx} className="bg-[#222222] rounded-lg border border-[#333333] overflow-hidden text-xs">
                          <button
                            onClick={() => setExpandedCitation(expandedCitation === cite.chunk_id ? null : cite.chunk_id)}
                            className="w-full p-2 text-left flex items-center justify-between hover:bg-[#2a2a2a] transition"
                          >
                            <span className="font-medium text-white truncate max-w-[70%]">
                              📄 {cite.title}
                            </span>
                            <div className="flex items-center gap-2 text-[10px]">
                              <span className="bg-[#10a37f]/10 text-[#10a37f] border border-[#10a37f]/30 px-1.5 py-0.5 rounded font-mono">
                                Match: {(cite.score * 100).toFixed(0)}%
                              </span>
                              <FiChevronDown className={`w-3.5 h-3.5 transition-transform ${expandedCitation === cite.chunk_id ? 'rotate-180' : ''}`} />
                            </div>
                          </button>

                          {expandedCitation === cite.chunk_id && (
                            <div className="p-2.5 border-t border-[#333333] bg-[#161616] text-[#b4b4b4] text-xs font-mono space-y-1">
                              <p className="text-[11px] text-[#10a37f] flex items-center gap-1">
                                <FiCornerDownRight /> Section: {cite.section_path || 'Root Document'}
                              </p>
                              {cite.text && (
                                <p className="bg-[#1f1f1f] p-2 rounded text-[#ececec] leading-normal font-sans border border-[#2a2a2a]">
                                  "{cite.text.slice(0, 300)}..."
                                </p>
                              )}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

              </div>
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>

        {/* ── FLOATING BOTTOM INPUT BAR (Matching Reference UI) ─────────────── */}
        <div className="p-4 md:px-12 lg:px-24 bg-[#111111]">
          <div className="max-w-3xl mx-auto bg-[#1e1e1e] border border-[#2a2a2a] rounded-full px-4 py-2 flex items-center gap-3 shadow-2xl focus-within:border-[#444444] transition">
            
            <button className="text-[#8e8e8e] hover:text-white transition">
              <FiPlus className="w-4 h-4" />
            </button>

            <input
              type="text"
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Send a Message"
              className="flex-1 bg-transparent text-white placeholder-[#666666] text-xs outline-none px-1"
            />

            {/* Model Selector Dropdown inside Input Bar */}
            <select
              value={provider}
              onChange={(e) => setProvider(e.target.value)}
              className="bg-transparent text-[#999999] hover:text-white font-mono text-[11px] outline-none cursor-pointer border-0 pr-1"
            >
              <option value="groq" className="bg-[#1e1e1e] text-white">llama-3.3-70b-versatile</option>
              <option value="gemini" className="bg-[#1e1e1e] text-white">gemini-3.0-flash</option>
            </select>

            <button
              onClick={() => handleSend()}
              disabled={!inputQuery.trim() || isLoading}
              className={`w-7 h-7 rounded-full transition flex items-center justify-center ${
                inputQuery.trim() && !isLoading
                  ? 'bg-white text-black hover:bg-gray-200 cursor-pointer shadow-md'
                  : 'bg-[#2a2a2a] text-[#555555] cursor-not-allowed'
              }`}
            >
              <FiSend className="w-3.5 h-3.5" />
            </button>

          </div>
        </div>

      </div>

      {/* ── PDF DOCUMENT KNOWLEDGE BASE MODAL ────────────────────────────────── */}
      {showDocModal && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#1c1c1c] border border-[#2c2c2c] rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#2c2c2c] pb-3">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <FiDatabase className="text-[#10a37f]" /> Knowledge Base Documents
              </h3>
              <button
                onClick={() => setShowDocModal(false)}
                className="p-1 rounded-md text-[#8e8e8e] hover:text-white hover:bg-[#2a2a2a]"
              >
                <FiX className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {documents.length === 0 ? (
                <p className="text-xs text-[#8e8e8e] text-center py-4">No documents indexed yet.</p>
              ) : (
                documents.map((doc, idx) => (
                  <div key={idx} className="p-3 bg-[#242424] border border-[#2c2c2c] rounded-xl flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2 truncate">
                      <FiFileText className="text-[#10a37f] shrink-0" />
                      <span className="font-medium text-white truncate">{doc.filename}</span>
                    </div>
                    <span className="text-[10px] bg-[#10a37f]/10 text-[#10a37f] px-2 py-0.5 rounded font-mono shrink-0">
                      {doc.chunk_count} Chunks
                    </span>
                  </div>
                ))
              )}
            </div>

            <button
              onClick={() => setShowDocModal(false)}
              className="w-full py-2 bg-[#2a2a2a] hover:bg-[#333333] border border-[#333333] rounded-xl text-white text-xs font-semibold transition"
            >
              Close
            </button>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;
