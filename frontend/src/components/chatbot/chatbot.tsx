import { useState } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import axios from "axios"
// import { UserRoundSearch } from 'lucide-react';

type Message = {
    id: number
    role: "user" | "bot"
    text: string
}
export default function Chatbot() {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState("")
    const [loading, setLoading] = useState(false)
    // const [data, setData] = useState<BotData>();

    async function handleSend() {
        if (!input.trim()) {
            return
        }
        const userMsg: Message = { id: Date.now(), role: "user", text: input }
        setMessages(prev => [...prev, userMsg])
        setInput("")
        setLoading(true)

        const user = {
            college_id : "iit_indore",
            query : userMsg.text,
            session_id : crypto.randomUUID()
        }
        try {
            const res = await axios.post(import.meta.env.VITE_BACKEND_URL+"api/ask", user, {
                headers: {
                    'Content-Type': 'application/json'
        }})
            const botMsg: Message = {
                id: Date.now(),
                role: "bot",
                text: res.data.answer,
            }
            setMessages(prev => [...prev, botMsg])
        }
        catch (err) {
            console.error(err)
        }
        finally {
            setLoading(false)
        }
    }

    return (
        <>
            <div className="px-3 bg-gradient-to-br from-[#4e1eeb] to-[#e0dced] rounded-2xl">
                <Card className="w-full max-w-lg mx-auto shadow-xl rounded-2xl bg-white">
                    <CardContent className="flex flex-col h-[500px]">
                        <ScrollArea className="flex-1 pr-2 space-y-2">
                            {messages.map(msg => (
                                <div
                                    key={msg.id}
                                    className={`p-3 rounded-xl max-w-[80%] my-1 ${msg.role === "user"
                                            ? "bg-[#4e1eeb] text-white ml-auto"
                                            : "bg-gray-200 text-black mr-auto"
                                        }`}
                                >
                                    {msg.text}
                                </div>
                            ))}
                            {loading && (
                                <div className="text-gray-500 text-sm italic">Typing...</div>
                            )}
                        </ScrollArea>

                        <div className="flex gap-2 mt-2">
                            <Input
                                placeholder="Type a message..."
                                value={input}
                                onChange={e => setInput(e.target.value)}
                                onKeyDown={e => e.key === "Enter" && handleSend()}
                                className="border-gray-400 border-2"
                            />
                            <Button onClick={handleSend} disabled={loading} className="bg-[#8b6cf1] active:bg-[#7952f7]">
                                Send
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>
        </>
    )
}
