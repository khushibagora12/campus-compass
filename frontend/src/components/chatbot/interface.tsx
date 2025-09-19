import { Bot } from "lucide-react"
import Chatbot from "./chatbot"

type ChatInterfaceProp = {
  toggle: boolean
  setToggle: React.Dispatch<React.SetStateAction<boolean>>
}

export default function ChatInterface({toggle, setToggle} : ChatInterfaceProp) {
    return (
        <>
            <div className="fixed top-5 right-5 border rounded-full p-3 cursor-pointer bg-gradient-to-tr from-violet-700 to-violet-300" onClick={() => setToggle(!toggle)}>
                <Bot color="white" />
                {/* <img src="/logo.png" alt="logo" className="h-10 w-10"/> */}
            </div>
            <div className={`absolute top-14 right-5 p-3 z-50 transition-all ${toggle ? 'opacity-100 scale-100' : 'opacity-0 scale-95 pointer-events-none'}`}>
                <Chatbot/>
            </div>
        </>
    )
}