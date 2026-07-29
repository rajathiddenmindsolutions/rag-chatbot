import { useState, useEffect, useRef } from 'react';
import { 
  FiMessageSquare, FiPlus, FiCpu, FiSend, FiFileText, 
  FiUploadCloud, FiLayers, FiDatabase, 
  FiChevronDown, FiChevronRight, FiMenu, FiX, FiCornerDownRight, FiSliders
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
  const [strategy, setStrategy] = useState<string>('semantic');
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
      setUploadStatus('❌ Upload failed (Server Error)');
    } finally {
      setUploading(false);
    }
  };

  const handleSend = async (queryText?: string) => {
    const textToSend = queryText || inputQuery;
    if (!textToSend.trim() || isLoading) return;

    const userMsgId = Date.now().toString();
    const userMsg: Message = {
      id: userMsgId,
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const botMsgId = (Date.now() + 1).toString();
    const botMsgPlaceholder: Message = {
      id: botMsgId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      provider: provider,
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

  const promptSuggestions = [
    { title: 'Data Preprocessing in Python', desc: 'Handling missing values, scaling & encoding techniques', icon: '🐍' },
    { title: 'Explain RAG Caching Architecture', desc: 'How Layer 1, Layer 2 & Layer 3 caching optimizes latency', icon: '⚡' },
    { title: 'LangGraph State Graph Engine', desc: 'How query routing, casual nodes & search tools function', icon: '🕸️' },
    { title: 'Search Google for AI News', desc: 'Retrieve latest real-time web news using Serper API', icon: '🌐' }
  ];

  return (
    <div className="flex h-screen bg-[#171717] text-[#ececec] overflow-hidden font-sans">
      
      {/* ── LEFT SIDEBAR (Open-WebUI Signature Style) ────────────────────────── */}
      <div 
        className={`fixed md:relative z-40 h-full bg-[#202123] border-r border-[#383838] flex flex-col transition-all duration-300 ${
          isSidebarOpen ? 'w-72' : 'w-0 -translate-x-full md:translate-x-0 md:w-0'
        }`}
      >
        {isSidebarOpen && (
          <div className="flex flex-col h-full p-3 space-y-3">
            
            {/* Logo & New Chat */}
            <div className="flex items-center justify-between px-2 pt-1 pb-2">
              <div className="flex items-center gap-2 font-bold text-lg text-white">
                <div className="w-7 h-7 rounded-lg bg-[#10a37f] flex items-center justify-center text-white shadow-md">
                  <FiCpu className="w-4 h-4" />
                </div>
                <span>Open-WebUI</span>
                <span className="text-[10px] bg-[#2f2f2f] text-[#10a37f] px-1.5 py-0.5 rounded font-mono border border-[#383838]">RAG</span>
              </div>
              <button 
                onClick={() => setIsSidebarOpen(false)}
                className="p-1.5 rounded-md text-[#8e8e8e] hover:text-white hover:bg-[#2f2f2f] transition"
              >
                <FiX className="w-5 h-5" />
              </button>
            </div>

            {/* New Chat Button */}
            <button
              onClick={startNewChat}
              className="flex items-center gap-3 w-full px-3 py-2.5 rounded-lg border border-[#383838] bg-[#262626] hover:bg-[#303030] text-white text-sm font-medium transition shadow-sm"
            >
              <FiPlus className="w-4 h-4 text-[#10a37f]" />
              <span>New Chat</span>
              <span className="ml-auto text-[10px] text-[#8e8e8e] border border-[#383838] px-1.5 py-0.5 rounded font-mono">⌘K</span>
            </button>

            {/* Knowledge Base Status */}
            <div className="bg-[#171717] rounded-lg p-2.5 border border-[#383838] text-xs">
              <div className="flex items-center justify-between text-[#b4b4b4] mb-1.5">
                <span className="flex items-center gap-1.5 font-semibold text-white">
                  <FiDatabase className="text-[#10a37f]" /> Vector Index
                </span>
                <span className="text-[10px] text-[#10a37f] bg-[#10a37f]/10 px-1.5 py-0.5 rounded font-mono">
                  {documents.length} Files
                </span>
              </div>
              <button 
                onClick={() => setShowDocModal(true)}
                className="w-full mt-1 px-2 py-1.5 rounded bg-[#262626] hover:bg-[#303030] border border-[#383838] text-[#ececec] flex items-center justify-between transition text-xs"
              >
                <span className="flex items-center gap-1.5 truncate">
                  <FiFileText className="text-[#b4b4b4]" /> Manage PDF Documents
                </span>
                <FiChevronRight className="w-3.5 h-3.5 text-[#8e8e8e]" />
              </button>
            </div>

            {/* Upload Drawer in Sidebar */}
            <div className="bg-[#262626] rounded-lg p-3 border border-[#383838] space-y-2">
              <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                <FiUploadCloud className="text-[#3b82f6]" /> Fast PDF Ingestion
              </span>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                className="block w-full text-xs text-[#b4b4b4] file:mr-2 file:py-1 file:px-2 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-[#10a37f] file:text-white hover:file:bg-[#1a7f64] cursor-pointer"
              />
              {uploadFile && (
                <button
                  onClick={handleFileUpload}
                  disabled={uploading}
                  className="w-full py-1.5 bg-[#10a37f] hover:bg-[#1a7f64] text-white text-xs font-medium rounded transition flex items-center justify-center gap-1.5"
                >
                  {uploading ? 'Processing & Indexing...' : 'Upload & Index PDF'}
                </button>
              )}
              {uploadStatus && (
                <p className="text-[11px] text-[#10a37f] mt-1 font-mono">{uploadStatus}</p>
              )}
            </div>

            {/* Conversation History List */}
            <div className="flex-1 overflow-y-auto space-y-1 pt-2">
              <span className="px-2 text-[11px] font-semibold text-[#8e8e8e] uppercase tracking-wider">Recent Conversations</span>
              <div 
                onClick={startNewChat}
                className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm text-[#ececec] bg-[#2f2f2f] hover:bg-[#383838] cursor-pointer transition border border-[#383838]"
              >
                <FiMessageSquare className="w-4 h-4 text-[#10a37f]" />
                <span className="truncate">Active RAG Session</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── MAIN CHAT VIEW AREA ──────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-[#171717]">
        
        {/* Top Header Controls Bar */}
        <div className="h-14 border-b border-[#383838] bg-[#171717]/90 backdrop-blur-md flex items-center justify-between px-4 z-30">
          <div className="flex items-center gap-3">
            {!isSidebarOpen && (
              <button 
                onClick={() => setIsSidebarOpen(true)}
                className="p-2 rounded-lg text-[#b4b4b4] hover:text-white hover:bg-[#2f2f2f] transition"
              >
                <FiMenu className="w-5 h-5" />
              </button>
            )}
            
            {/* Open-WebUI Model Selector Dropdown */}
            <div className="relative flex items-center bg-[#262626] border border-[#383838] rounded-lg px-2.5 py-1.5 text-xs text-white font-medium hover:border-[#10a37f] transition shadow-sm">
              <FiCpu className="text-[#10a37f] mr-2" />
              <select
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                className="bg-transparent text-white font-semibold outline-none cursor-pointer pr-4 appearance-none text-xs"
              >
                <option value="groq" className="bg-[#262626] text-white">⚡ Open-WebUI • Groq (LLaMA-3.3-70B)</option>
                <option value="gemini" className="bg-[#262626] text-white">✨ Open-WebUI • Google Gemini 3.0 Flash</option>
              </select>
              <FiChevronDown className="w-3.5 h-3.5 text-[#8e8e8e] pointer-events-none absolute right-2" />
            </div>

            {/* Chunking Strategy Pill Selector */}
            <div className="hidden sm:flex items-center bg-[#262626] border border-[#383838] rounded-lg px-2 py-1 text-xs text-[#b4b4b4] gap-1.5">
              <FiSliders className="text-[#3b82f6]" />
              <span className="text-[11px] text-[#8e8e8e]">Strategy:</span>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="bg-transparent text-white font-medium outline-none cursor-pointer text-xs"
              >
                <option value="structural" className="bg-[#262626]">Structural Chunking</option>
                <option value="semantic" className="bg-[#262626]">Semantic Chunking</option>
                <option value="hierarchical" className="bg-[#262626]">Hierarchical Chunking</option>
              </select>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <span className="hidden sm:inline-block text-[11px] bg-[#10a37f]/10 text-[#10a37f] border border-[#10a37f]/30 px-2 py-0.5 rounded-full font-mono">
              ● RAG Pipeline Active
            </span>
          </div>
        </div>

        {/* Chat Messages Scroll Container */}
        <div className="flex-1 overflow-y-auto px-4 md:px-12 lg:px-24 py-6 space-y-6">
          
          {/* Empty State Hero (Open-WebUI Signature Landing Page) */}
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[70vh] text-center max-w-2xl mx-auto space-y-6">
              <div className="w-16 h-16 rounded-2xl bg-[#10a37f] flex items-center justify-center text-white text-3xl shadow-xl shadow-[#10a37f]/20 border border-[#10a37f]/40">
                <FiCpu />
              </div>
              <div>
                <h1 className="text-2xl md:text-3xl font-bold text-white tracking-tight">
                  What would you like to explore today?
                </h1>
                <p className="text-sm text-[#b4b4b4] mt-2">
                  Ask anything across your PDF Knowledge Base or real-time web data powered by LangGraph RAG.
                </p>
              </div>

              {/* Prompt Suggestions Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full pt-4">
                {promptSuggestions.map((item, idx) => (
                  <button
                    key={idx}
                    onClick={() => handleSend(item.title)}
                    className="p-3.5 text-left rounded-xl bg-[#202123] border border-[#383838] hover:border-[#10a37f] hover:bg-[#262626] transition group shadow-sm flex flex-col justify-between"
                  >
                    <div className="flex items-center gap-2 text-white font-medium text-sm">
                      <span className="text-base">{item.icon}</span>
                      <span className="group-hover:text-[#10a37f] transition">{item.title}</span>
                    </div>
                    <span className="text-xs text-[#8e8e8e] mt-1 line-clamp-1">{item.desc}</span>
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
                <div className="w-8 h-8 rounded-full bg-[#10a37f] flex items-center justify-center text-white text-sm font-bold shadow-md shrink-0 mt-1">
                  <FiCpu />
                </div>
              )}

              {/* Message Box */}
              <div className={`flex flex-col max-w-[85%] space-y-2`}>
                
                {/* Header Role info */}
                <div className="flex items-center gap-2 text-[11px] text-[#8e8e8e]">
                  <span className="font-semibold text-[#ececec]">
                    {msg.role === 'user' ? 'You' : `Open-WebUI (${provider === 'gemini' ? 'Gemini 3.0' : 'Groq LLaMA-3.3'})`}
                  </span>
                  <span>•</span>
                  <span>{msg.timestamp}</span>
                </div>

                {/* Message Bubble Container */}
                <div
                  className={`p-4 rounded-2xl text-sm leading-relaxed ${
                    msg.role === 'user'
                      ? 'bg-[#2f2f2f] text-white border border-[#383838] rounded-tr-none'
                      : 'bg-[#202123] text-[#ececec] border border-[#383838] rounded-tl-none shadow-sm'
                  }`}
                >
                  {msg.role === 'user' ? (
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                  ) : (
                    <div className="markdown-content">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {msg.content || '▌'}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>

                {/* Open-WebUI Citation Inspector Accordion */}
                {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
                  <div className="mt-3 bg-[#1e1e1e] border border-[#383838] rounded-xl p-3 space-y-2">
                    <div className="flex items-center justify-between text-xs font-semibold text-[#10a37f]">
                      <span className="flex items-center gap-1.5">
                        <FiLayers /> Source Inspector ({msg.citations.length} Verified Chunks)
                      </span>
                    </div>

                    <div className="space-y-1.5 pt-1">
                      {msg.citations.map((cite, cIdx) => (
                        <div key={cIdx} className="bg-[#262626] rounded-lg border border-[#383838] overflow-hidden text-xs">
                          <button
                            onClick={() => setExpandedCitation(expandedCitation === cite.chunk_id ? null : cite.chunk_id)}
                            className="w-full p-2 text-left flex items-center justify-between hover:bg-[#303030] transition"
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
                            <div className="p-2.5 border-t border-[#383838] bg-[#1a1a1a] text-[#b4b4b4] text-xs font-mono space-y-1">
                              <p className="text-[11px] text-[#10a37f] flex items-center gap-1">
                                <FiCornerDownRight /> Section: {cite.section_path || 'Root Document'}
                              </p>
                              {cite.text && (
                                <p className="bg-[#222] p-2 rounded text-[#ececec] leading-normal font-sans border border-[#333]">
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

        {/* ── FLOATING BOTTOM INPUT BAR (Open-WebUI Signature Style) ─────────────── */}
        <div className="p-4 md:px-12 lg:px-24 bg-[#171717] border-t border-[#383838]/40">
          <div className="max-w-3xl mx-auto relative bg-[#2f2f2f] border border-[#383838] rounded-2xl p-2.5 shadow-xl focus-within:border-[#10a37f] transition">
            
            <textarea
              value={inputQuery}
              onChange={(e) => setInputQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Ask anything or search your PDF documents..."
              rows={1}
              className="w-full bg-transparent text-white placeholder-[#8e8e8e] text-sm outline-none resize-none px-2 py-1 max-h-32 min-h-[40px]"
            />

            {/* Input Bar Controls */}
            <div className="flex items-center justify-between pt-2 border-t border-[#383838]/60 mt-1">
              
              <div className="flex items-center gap-2 text-xs text-[#b4b4b4]">
                <button
                  onClick={() => setShowDocModal(true)}
                  className="p-1.5 hover:bg-[#383838] rounded-lg text-[#8e8e8e] hover:text-white transition flex items-center gap-1"
                  title="PDF Knowledge Base"
                >
                  <FiFileText className="w-4 h-4" />
                  <span className="text-[11px] hidden sm:inline">Knowledge</span>
                </button>
              </div>

              {/* Send Button */}
              <button
                onClick={() => handleSend()}
                disabled={!inputQuery.trim() || isLoading}
                className={`p-2 rounded-xl text-white transition flex items-center justify-center ${
                  inputQuery.trim() && !isLoading
                    ? 'bg-[#10a37f] hover:bg-[#1a7f64] shadow-md cursor-pointer'
                    : 'bg-[#383838] text-[#8e8e8e] cursor-not-allowed'
                }`}
              >
                <FiSend className="w-4 h-4" />
              </button>

            </div>
          </div>

          <p className="text-center text-[10px] text-[#8e8e8e] mt-2 font-mono">
            Open-WebUI RAG • Powered by OpenSearch, LangGraph & Multi-Model LLM Engine
          </p>
        </div>

      </div>

      {/* ── PDF DOCUMENT KNOWLEDGE BASE MODAL ────────────────────────────────── */}
      {showDocModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#202123] border border-[#383838] rounded-2xl max-w-lg w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#383838] pb-3">
              <h3 className="text-lg font-bold text-white flex items-center gap-2">
                <FiDatabase className="text-[#10a37f]" /> Knowledge Base Documents
              </h3>
              <button
                onClick={() => setShowDocModal(false)}
                className="p-1 rounded-md text-[#8e8e8e] hover:text-white hover:bg-[#2f2f2f]"
              >
                <FiX className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
              {documents.length === 0 ? (
                <p className="text-xs text-[#8e8e8e] text-center py-4">No documents indexed yet.</p>
              ) : (
                documents.map((doc, idx) => (
                  <div key={idx} className="p-3 bg-[#262626] border border-[#383838] rounded-xl flex items-center justify-between text-xs">
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
              className="w-full py-2 bg-[#2f2f2f] hover:bg-[#383838] border border-[#383838] rounded-xl text-white text-xs font-semibold transition"
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
